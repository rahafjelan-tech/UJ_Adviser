from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nbformat
import sqlite3
import os
import traceback

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

NOTEBOOK_PATH = "UJAD1 (1).ipynb"
DB_PATH       = "student_data.db"
global_scope  = {}


# =========================
# Notebook loader
# =========================
def load_notebook():
    global global_scope

    if not os.path.exists(NOTEBOOK_PATH):
        print(f"الملف {NOTEBOOK_PATH} غير موجود.")
        global_scope = {}
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    global_scope = {}

    for cell in nb.cells:
        if cell.cell_type == "code":
            try:
                lines = cell.source.splitlines()
                cleaned_lines = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("!") or stripped.startswith("%"):
                        continue
                    cleaned_lines.append(line)
                cleaned_code = "\n".join(cleaned_lines).strip()
                if not cleaned_code:
                    continue
                exec(cleaned_code, global_scope)
            except Exception as e:
                print("خطأ في تنفيذ خلية من الـ notebook:", e)

    # PATCH 1: dense_topk — fault-isolate broken HNSW collections
    _orig_dense_topk = global_scope.get("dense_topk")
    if _orig_dense_topk is not None:
        def _safe_dense_topk(collection, qvec, k, _orig=_orig_dense_topk):
            try:
                return _orig(collection, qvec, k)
            except Exception as e:
                if any(kw in str(e).lower() for kw in [
                    "hnsw", "nothing found on disk", "segment", "sqlite",
                    "no such table", "disk", "storage"
                ]):
                    print(f"[PATCH] Skipping broken collection '{collection}': {e}")
                    return []
                raise
        global_scope["dense_topk"] = _safe_dense_topk
        print("[PATCH] dense_topk wrapped with HNSW fault-isolation.")

    # PATCH 2: bm25_build
    _orig_bm25_build = global_scope.get("bm25_build")
    if _orig_bm25_build is not None:
        def _safe_bm25_build(collection, _orig=_orig_bm25_build):
            try:
                return _orig(collection)
            except Exception as e:
                if any(kw in str(e).lower() for kw in [
                    "hnsw", "nothing found on disk", "segment", "sqlite",
                    "no such table", "disk", "storage"
                ]):
                    print(f"[PATCH] Skipping broken BM25 collection '{collection}': {e}")
                    return (None, [])
                raise
        global_scope["bm25_build"] = _safe_bm25_build
        print("[PATCH] bm25_build wrapped with HNSW fault-isolation.")

    # PATCH 3: bm25_topk
    _orig_bm25_topk = global_scope.get("bm25_topk")
    if _orig_bm25_topk is not None:
        def _safe_bm25_topk(collection, query, k, _orig=_orig_bm25_topk):
            try:
                return _orig(collection, query, k)
            except Exception as e:
                if any(kw in str(e).lower() for kw in [
                    "hnsw", "nothing found on disk", "segment", "sqlite",
                    "no such table", "disk", "storage", "nonetype"
                ]):
                    print(f"[PATCH] Skipping broken BM25 topk '{collection}': {e}")
                    return []
                raise
        global_scope["bm25_topk"] = _safe_bm25_topk
        print("[PATCH] bm25_topk wrapped with HNSW fault-isolation.")

    # PATCH 4: REMOVED — caused double-save in advisor path.
    # execute_single_question already calls save_turn_to_memory() at the end.
    # Wrapping answer() here also saves, which duplicates history and corrupts
    # the memory rewrite on the second question.
    print("[PATCH 4] Skipped intentionally — save_turn handled by execute_single_question.")

    # PATCH 5: normalize_sql_text — used directly in Cell 13 (_match_course_from_state)
    # but never defined anywhere in the notebook. Inject a safe fallback.
    if "normalize_sql_text" not in global_scope:
        import re as _re
        def _normalize_sql_text(text: str) -> str:
            if not text:
                return ""
            t = str(text).strip()
            t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            t = t.replace("ة", "ه").replace("ى", "ي")
            t = _re.sub(r"\s+", " ", t)
            return t
        global_scope["normalize_sql_text"] = _normalize_sql_text
        print("[PATCH 5] normalize_sql_text injected.")

    # PATCH 6: make student answer save once.
    _orig_answer_s = global_scope.get("answer")
    _save_turn_s   = global_scope.get("save_turn_to_memory")
    if _orig_answer_s is not None and _save_turn_s is not None:
        def _student_answer_with_save(query, protos, mode="router",
                                      return_debug=False, use_deep=False,
                                      use_web=False,
                                      _orig=_orig_answer_s, _save=_save_turn_s):
            result = _orig(
                query=query, protos=protos, mode=mode,
                return_debug=return_debug,
                use_deep=use_deep, use_web=use_web,
            )
            if not return_debug and not use_deep:
                try:
                    answer_text = result if isinstance(result, str) else result.get("answer", "")
                    _save(query, answer_text)
                except Exception as e:
                    print(f"[PATCH 6] save_turn_to_memory failed: {e}")
            return result
        global_scope["answer"] = _student_answer_with_save
        print("[PATCH 6] answer() wrapped for student-path save only.")

    # PATCH 7: Normalize all retrieval context_text values to strings.
    # Root cause fix for: object of type 'int' has no len()
    def _safe_text(value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            return "\n".join(str(x) for x in value if x is not None)
        if isinstance(value, dict):
            return str(value)
        return str(value)

    def _normalize_retrieval_result(result):
        if isinstance(result, dict):
            result["context_text"] = _safe_text(result.get("context_text", ""))
            result["sources"] = result.get("sources") or []
        return result

    _orig_context_answers_question = global_scope.get("context_answers_question")
    if _orig_context_answers_question is not None:
        def _safe_context_answers_question(question, context_text, _orig=_orig_context_answers_question):
            return _orig(_safe_text(question), _safe_text(context_text))

        global_scope["context_answers_question"] = _safe_context_answers_question
        print("[PATCH 7A] context_answers_question made string-safe.")

    _orig_retrieve_final = global_scope.get("retrieve_final")
    if _orig_retrieve_final is not None:
        def _safe_retrieve_final(*args, _orig=_orig_retrieve_final, **kwargs):
            result = _orig(*args, **kwargs)
            return _normalize_retrieval_result(result)

        global_scope["retrieve_final"] = _safe_retrieve_final
        print("[PATCH 7B] retrieve_final wrapped with context normalization.")

    if "retrieve_final" in global_scope:
        def _safe_retrieve_with_router_as_hint(
            query,
            protos,
            mode="router",
            route_decision=None,
            use_bm25=True
        ):
            retrieve_final = global_scope["retrieve_final"]
            context_answers_question = global_scope["context_answers_question"]

            routed_result = retrieve_final(
                query=query,
                protos=protos,
                mode=mode,
                use_bm25=use_bm25
            )

            routed_result = _normalize_retrieval_result(routed_result)
            routed_context = _safe_text(routed_result.get("context_text", ""))
            routed_sources = routed_result.get("sources", []) or []
            routed_answered = context_answers_question(query, routed_context)

            routed_result["retrieval_scope"] = "router_hint"
            routed_result["router_hint_answered"] = routed_answered
            routed_result["router_was_condition"] = False

            if routed_answered:
                routed_result["expanded_to_all_collections"] = False
                return routed_result

            try:
                all_result = retrieve_final(
                    query=query,
                    protos=protos,
                    mode="all",
                    use_bm25=use_bm25
                )

                all_result = _normalize_retrieval_result(all_result)
                all_context = _safe_text(all_result.get("context_text", ""))
                all_sources = all_result.get("sources", []) or []
                all_answered = context_answers_question(query, all_context)

                all_result["retrieval_scope"] = "all_collections"
                all_result["router_hint_answered"] = routed_answered
                all_result["all_collections_answered"] = all_answered
                all_result["expanded_to_all_collections"] = True
                all_result["router_was_condition"] = False
                all_result["router_hint_result"] = {
                    "context_text": routed_context,
                    "sources": routed_sources,
                }

                if all_answered:
                    return all_result

                if len(all_context) > len(routed_context):
                    return all_result

            except Exception as e:
                routed_result["all_collections_error"] = str(e)

            routed_result["expanded_to_all_collections"] = False
            return routed_result

        global_scope["retrieve_with_router_as_hint"] = _safe_retrieve_with_router_as_hint
        print("[PATCH 7C] retrieve_with_router_as_hint replaced with string-safe version.")

    _orig_kb_agent_search_deep = global_scope.get("kb_agent_search_deep")
    if _orig_kb_agent_search_deep is not None:
        def _safe_kb_agent_search_deep(*args, _orig=_orig_kb_agent_search_deep, **kwargs):
            result = _orig(*args, **kwargs)

            if isinstance(result, dict):
                return _normalize_retrieval_result(result)

            if hasattr(result, "context_text"):
                try:
                    result.context_text = _safe_text(result.context_text)
                except Exception:
                    pass

            if hasattr(result, "sources"):
                try:
                    result.sources = result.sources or []
                except Exception:
                    pass

            return result

        global_scope["kb_agent_search_deep"] = _safe_kb_agent_search_deep
        print("[PATCH 7D] kb_agent_search_deep wrapped with context normalization.")

    print("[app.py] All patches applied successfully.")


load_notebook()

if "ai_autocorrect" not in global_scope:
    def ai_autocorrect(text):
        return text
    global_scope["ai_autocorrect"] = ai_autocorrect
    print("[PATCH] ai_autocorrect fallback added.")


# =========================
# Fallback patches for Render
# =========================
if "basic_autocorrect" not in global_scope:
    def basic_autocorrect(text):
        return text
    global_scope["basic_autocorrect"] = basic_autocorrect
    print("[PATCH] basic_autocorrect fallback added.")


if "client" not in global_scope:
    try:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            global_scope["client"] = OpenAI(api_key=api_key)
            print("[PATCH] OpenAI client added from environment.")
        else:
            print("[WARNING] OPENAI_API_KEY is missing.")
    except Exception as e:
        print("[WARNING] Could not create OpenAI client:", e)


# =========================
# DB helper
# =========================
def _dict_rows(cursor):
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


# =========================
# HTML pages
# =========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/student-page")
def student_page():
    return render_template("student.html")


@app.route("/advisor-page")
def advisor_page():
    return render_template("advisor.html")


@app.route("/employee-page")
def employee_page():
    return render_template("employee.html")


# =========================
# Health check
# =========================
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "notebook_loaded": bool(global_scope),
        "db_available": os.path.exists(DB_PATH),
        "has_answer": "answer" in global_scope,
        "has_rag_student_answer": "rag_student_answer" in global_scope,
        "has_protos": "protos" in global_scope
    })


# =========================
# Student API
# =========================
@app.route("/student", methods=["POST"])
def student():
    data     = request.get_json(silent=True) or {}
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"reply": "الرجاء إدخال سؤال."}), 400

    if "answer" not in global_scope:
        return jsonify({"reply": "دالة answer غير موجودة في notebook"}), 500

    try:
        protos = global_scope.get("protos", None)
        result = global_scope["answer"](
            query=question,
            protos=protos,
            mode="router",
            return_debug=False
        )

        if isinstance(result, dict):
            return jsonify({"reply": result.get("answer", "لا يوجد رد")})

        return jsonify({"reply": str(result)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": f"خطأ في معالجة سؤال الطالب: {str(e)}"}), 500


# =========================
# Advisor chat API
# =========================
@app.route("/advisor", methods=["POST"])
def advisor():
    data = request.get_json(silent=True) or {}
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"reply": "الرجاء إدخال سؤال."}), 400

    normalized_q = question.replace("؟", "").replace("?", "").strip().lower()

    identity_questions = [
        "من انت",
        "من أنت",
        "مين انت",
        "مين أنت",
        "ما اسمك",
        "وش اسمك",
        "عرفني بنفسك"
    ]

    if normalized_q in identity_questions:
        return jsonify({
            "reply": "أنا مساعد المرشد الأكاديمي. أستطيع مساعدتك في الاستفسارات الأكاديمية، بيانات الطلاب، الحالات المتعثرة، التحصيل، الحضور، والخطط الدراسية."
        })

    if "rag_student_answer" not in global_scope:
        return jsonify({"reply": "دالة rag_student_answer غير موجودة"}), 500

    try:
        protos = global_scope.get("protos", None)

        result = global_scope["rag_student_answer"](
            question=question,
            protos=protos,
            retrieval_mode="router"
        )

        if isinstance(result, dict):
            return jsonify({"reply": result.get("answer", "لا يوجد رد")})

        return jsonify({"reply": str(result)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": f"خطأ في معالجة سؤال المرشد: {str(e)}"}), 500


# =========================
# Advisor Dashboard API  ← NEW (from ZIP)
# =========================
@app.route("/advisor-dashboard-data")
def advisor_dashboard_data():
    advisor_name = request.args.get("advisor", "ريم المطيري").strip() or "ريم المطيري"

    if not os.path.exists(DB_PATH):
        return jsonify({"error": "قاعدة بيانات الطلاب غير موجودة داخل مجلد المشروع."}), 500

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()

        total_students = cur.execute(
            "SELECT COUNT(*) FROM students WHERE advisor = ?",
            (advisor_name,)
        ).fetchone()[0]

        metrics = cur.execute("""
            SELECT
              ROUND(AVG(r.attendance), 1),
              ROUND(AVG((r.attendance + r.participation + r.quizzes +
                         r.assignments + r.midterm + r.projects) / 6.0), 1),
              SUM(CASE WHEN s.graduating_this_year = 'Y' THEN 1 ELSE 0 END),
              SUM(CASE WHEN s.has_medical_condition = 1 THEN 1 ELSE 0 END),
              SUM(CASE WHEN s.needs_accommodation = 1 THEN 1 ELSE 0 END),
              SUM(CASE WHEN ((r.attendance + r.participation + r.quizzes +
                              r.assignments + r.midterm + r.projects) / 6.0) < 60
                        THEN 1 ELSE 0 END)
            FROM students s
            LEFT JOIN risk_features r ON s.student_id = r.student_id
            WHERE s.advisor = ?
        """, (advisor_name,)).fetchone()

        top_risk = _dict_rows(cur.execute("""
            SELECT s.student_id, s.student_name, s.hours_completed,
                   s.remaining_hours, ROUND(r.attendance, 1) AS attendance,
                   ROUND((r.attendance + r.participation + r.quizzes +
                          r.assignments + r.midterm + r.projects) / 6.0, 1) AS risk_score
            FROM students s
            LEFT JOIN risk_features r ON s.student_id = r.student_id
            WHERE s.advisor = ?
            ORDER BY risk_score ASC
            LIMIT 5
        """, (advisor_name,)))

        support = _dict_rows(cur.execute("""
            SELECT s.student_id, s.student_name,
                   sf.mental_stress_level, sf.financial_stress,
                   sf.family_support, sf.study_hours_per_week,
                   sf.sleep_duration_hours_per_night
            FROM students s
            LEFT JOIN support_features sf ON s.student_id = sf.student_id
            WHERE s.advisor = ?
            ORDER BY sf.mental_stress_level DESC, sf.financial_stress DESC
            LIMIT 5
        """, (advisor_name,)))

        courses = _dict_rows(cur.execute("""
            SELECT sc.course_key, sc.course_label, COUNT(*) AS repeat_count
            FROM student_courses sc
            JOIN students s ON sc.student_id = s.student_id
            WHERE s.advisor = ? AND sc.status = 'R'
            GROUP BY sc.course_key, sc.course_label
            ORDER BY repeat_count DESC
            LIMIT 5
        """, (advisor_name,)))

        return jsonify({
            "advisor": advisor_name,
            "total_students": total_students,
            "avg_attendance": metrics[0] or 0,
            "avg_performance": metrics[1] or 0,
            "graduating_this_year": metrics[2] or 0,
            "medical_cases": metrics[3] or 0,
            "accommodation_cases": metrics[4] or 0,
            "high_risk_students": metrics[5] or 0,
            "top_risk_students": top_risk,
            "support_students": support,
            "repeated_courses": courses,
        })
    finally:
        conn.close()


# =========================
# Debug
# =========================
@app.route("/debug-scope")
def debug_scope():
    return jsonify({
        "loaded_names": list(global_scope.keys()),
        "has_answer": "answer" in global_scope,
        "has_global_student_answering": "global_student_answering" in global_scope,
        "has_protos": "protos" in global_scope,
        "has_vectorstore": "vectorstore" in global_scope,
        "has_retriever": "retriever" in global_scope,
        "has_documents": "documents" in global_scope,
        "has_chunks": "chunks" in global_scope,
        "has_collection": "collection" in global_scope,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
