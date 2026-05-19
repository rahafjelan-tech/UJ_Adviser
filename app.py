from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import ast
import nbformat
import sqlite3
import os
import re
import traceback
from pathlib import Path

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

BASE_DIR      = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "UJAD1 (1).ipynb"
DB_PATH       = BASE_DIR / "student_data.db"
global_scope  = {}
notebook_error = None
notebook_warnings = []

# Render/gunicorn can start the app from a different current working directory.
# The notebook contains many relative data paths, so anchor execution at the repo.
os.chdir(BASE_DIR)

NOTEBOOK_PATH_REPLACEMENTS = {
    '"data/vector_database_updated.zip"': '"data/vector_database_updated (1).zip"',
    "chroma.list_collections()": "_safe_chroma_list_collections(chroma)",
    "chroma = chromadb.PersistentClient(path=PERSIST_DIR)": (
        "_ensure_chroma_schema_compat(CHROMA_DIR)\n"
        "chroma = chromadb.PersistentClient(path=PERSIST_DIR)"
    ),
}


def _clean_notebook_source(source):
    lines = []
    for line in (source or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!") or stripped.startswith("%"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _is_demo_stmt(stmt):
    if isinstance(stmt, ast.Expr):
        if isinstance(stmt.value, ast.Name):
            return True
        if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
            return stmt.value.func.id in {
                "answer",
                "rag_student_answer",
                "global_student_answering",
                "sql_student_answering",
                "run_system_test",
            }

    if isinstance(stmt, ast.Assign):
        target_names = {
            target.id for target in stmt.targets if isinstance(target, ast.Name)
        }
        if target_names and target_names.issubset({
            "rag_result",
            "student_result",
            "test_result",
            "test_results",
            "test_questions",
            "test_questions1",
        }):
            return True
        if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
            return stmt.value.func.id in {
                "answer",
                "rag_student_answer",
                "global_student_answering",
                "sql_student_answering",
                "run_system_test",
            }

    return False


def _prepare_notebook_code(source):
    cleaned_code = _clean_notebook_source(source)
    if not cleaned_code:
        return None, "empty/shell-only"

    for old, new in NOTEBOOK_PATH_REPLACEMENTS.items():
        cleaned_code = cleaned_code.replace(old, new)

    try:
        tree = ast.parse(cleaned_code, mode="exec")
    except SyntaxError:
        return compile(cleaned_code, "<notebook-cell>", "exec"), ""

    kept = [stmt for stmt in tree.body if not _is_demo_stmt(stmt)]
    removed = len(tree.body) - len(kept)

    if not kept:
        return None, "demo/display-only"

    if removed:
        tree = ast.Module(body=kept, type_ignores=[])
        ast.fix_missing_locations(tree)
        return compile(tree, "<notebook-cell>", "exec"), "demo statements removed"

    return compile(cleaned_code, "<notebook-cell>", "exec"), ""


def _ensure_chroma_schema_compat(db_dir):
    base = Path(db_dir)

    candidates = []

    direct = base / "chroma.sqlite3"
    if direct.exists():
        candidates.append(direct)

    if base.exists():
        candidates.extend(base.rglob("chroma.sqlite3"))

    # remove duplicates, preserve order
    seen = set()
    db_paths = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            db_paths.append(p)

    if not db_paths:
        print(f"[PATCH] No chroma.sqlite3 found under {base}")
        return

    for db_path in db_paths:
        print(f"[PATCH] Checking Chroma sqlite schema: {db_path}")

        conn = sqlite3.connect(db_path)
        try:
            def _table_exists(table_name):
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                ).fetchone()
                return row is not None

            def _ensure_column(table_name, column_name, column_type="TEXT"):
                if not _table_exists(table_name):
                    print(f"[PATCH] Chroma table missing in {db_path}: {table_name}")
                    return

                columns = {
                    row[1]
                    for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                }

                if column_name not in columns:
                    conn.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                    )
                    print(
                        f"[PATCH] Added missing Chroma column "
                        f"{table_name}.{column_name} in {db_path}"
                    )

            _ensure_column("collections", "topic", "TEXT")
            _ensure_column("segments", "topic", "TEXT")

            conn.commit()

        finally:
            conn.close()

class _CollectionRef:
    def __init__(self, name):
        self.name = name


def _safe_chroma_list_collections(client):
    try:
        return client.list_collections()
    except Exception as e:
        if "collections.topic" not in str(e):
            raise

        expected = global_scope.get("ALL_COLLECTIONS") or [
            "transfer_rules",
            "regulations",
            "degree_plans",
            "student_helpers",
            "faculty_offices",
            "academic_calendar",
        ]

        available = []
        for name in expected:
            try:
                client.get_collection(name)
                available.append(_CollectionRef(name))
            except Exception:
                pass

        print(
            "[PATCH] chroma.list_collections() schema mismatch "
            f"({e}); using get_collection fallback: {[c.name for c in available]}"
        )
        return available

# =========================
# Notebook loader
# =========================
def load_notebook():
    global global_scope, notebook_error, notebook_warnings
    notebook_error = None
    notebook_warnings = []

    if not NOTEBOOK_PATH.exists():
        notebook_error = f"Notebook file not found: {NOTEBOOK_PATH}"
        print(notebook_error)
        global_scope = {}
        return

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    global_scope = {
        "_safe_chroma_list_collections": _safe_chroma_list_collections,
        "_ensure_chroma_schema_compat": _ensure_chroma_schema_compat,
    }

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            try:
                code_obj, reason = _prepare_notebook_code(cell.source)
                if code_obj is None:
                    notebook_warnings.append(f"Cell {idx} skipped: {reason}")
                    continue
                if reason:
                    notebook_warnings.append(f"Cell {idx}: {reason}")

                exec(code_obj, global_scope)
            except Exception as e:
                notebook_error = f"Notebook failed while loading cell {idx}: {e}"
                print("=" * 80)
                print(f"[NOTEBOOK LOAD ERROR] Cell index: {idx}")
                print(f"[NOTEBOOK LOAD ERROR] Exception: {type(e).__name__}: {e}")
                print("[CELL SOURCE START]")
                print(cell.source[:3000])
                print("[CELL SOURCE END]")
                traceback.print_exc()
                print("=" * 80)
                global_scope = {}
                return

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
                    print(f"[RAG ERROR] dense_topk failed for collection '{collection}': {type(e).__name__}: {e}")
                    raise
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
                    print(f"[RAG ERROR] bm25_build failed for collection '{collection}': {type(e).__name__}: {e}")
                    raise
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
                    print(f"[RAG ERROR] bm25_topk failed for collection '{collection}': {type(e).__name__}: {e}")
                    raise
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
                use_deep=use_deep,
                use_web=use_web,
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

    # PATCH 7: Global safety wrappers for non-SQL RAG answers.
    # Fixes: object of type 'int' has no len()
    def _safe_text(value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            return "\n".join(str(x) for x in value if x is not None)
        return str(value)

    def _safe_sources(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (tuple, set)):
            return list(value)
        return [str(value)]

    def _normalize_retrieval_result(result):
        if isinstance(result, dict):
            result["context_text"] = _safe_text(result.get("context_text", ""))
            result["sources"] = _safe_sources(result.get("sources", []))
        return result

    _orig_context_answers_question = global_scope.get("context_answers_question")
    if _orig_context_answers_question is not None:
        def _safe_context_answers_question(question, context_text, _orig=_orig_context_answers_question):
            return _orig(_safe_text(question), _safe_text(context_text))
        global_scope["context_answers_question"] = _safe_context_answers_question
        print("[PATCH 7A] context_answers_question is string-safe.")

    _orig_retrieve_final = global_scope.get("retrieve_final")
    if _orig_retrieve_final is not None:
        def _safe_retrieve_final(*args, _orig=_orig_retrieve_final, **kwargs):
            result = _orig(*args, **kwargs)
            return _normalize_retrieval_result(result)
        global_scope["retrieve_final"] = _safe_retrieve_final
        print("[PATCH 7B] retrieve_final is string-safe.")

    _orig_llm_rag_answer = global_scope.get("llm_rag_answer_from_contexts")
    if _orig_llm_rag_answer is not None:
        def _safe_llm_rag_answer_from_contexts(
            question,
            kb_context="",
            uj_context="",
            web_context="",
            sources_kb=None,
            sources_uj=None,
            sources_web=None,
            _orig=_orig_llm_rag_answer
        ):
            return _orig(
                question=_safe_text(question),
                kb_context=_safe_text(kb_context),
                uj_context=_safe_text(uj_context),
                web_context=_safe_text(web_context),
                sources_kb=_safe_sources(sources_kb),
                sources_uj=_safe_sources(sources_uj),
                sources_web=_safe_sources(sources_web),
            )
        global_scope["llm_rag_answer_from_contexts"] = _safe_llm_rag_answer_from_contexts
        print("[PATCH 7C] llm_rag_answer_from_contexts is string-safe.")

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
                    result.sources = _safe_sources(result.sources)
                except Exception:
                    pass

            return result

        global_scope["kb_agent_search_deep"] = _safe_kb_agent_search_deep
        print("[PATCH 7D] kb_agent_search_deep is string-safe.")

    # PATCH 8: Emergency advisor fallback.
    # If rag_student_answer crashes, retry using answer() with deep search disabled.
    _orig_rag_student_answer = global_scope.get("rag_student_answer")
    _orig_answer_for_fallback = global_scope.get("answer")

    if _orig_rag_student_answer is not None and _orig_answer_for_fallback is not None:
        def _safe_rag_student_answer(question, protos=None, retrieval_mode="router",
                                     _orig_rag=_orig_rag_student_answer,
                                     _fallback_answer=_orig_answer_for_fallback):
            try:
                return _orig_rag(
                    question=_safe_text(question),
                    protos=protos,
                    retrieval_mode=retrieval_mode
                )
            except TypeError as e:
                if "has no len" not in str(e):
                    raise

                print(f"[PATCH 8] rag_student_answer failed with len(int). Retrying via safe answer(): {e}")

                fallback = _fallback_answer(
                    query=_safe_text(question),
                    protos=protos,
                    mode="all",
                    return_debug=False,
                    use_deep=False,
                    use_web=True
                )

                if isinstance(fallback, dict):
                    return fallback

                return {"answer": _safe_text(fallback)}

        global_scope["rag_student_answer"] = _safe_rag_student_answer
        print("[PATCH 8] rag_student_answer wrapped with safe fallback.")

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


def _chroma_health():
    client = global_scope.get("chroma")
    expected = global_scope.get("ALL_COLLECTIONS") or []

    result = {
        "available": client is not None,
        "count": 0,
        "collections": [],
        "expected_count": len(expected),
        "expected_collections": expected,
        "missing_expected": expected,
        "error": None,
    }

    if client is None:
        return result

    try:
        collections = _safe_chroma_list_collections(client)
        names = [getattr(col, "name", str(col)) for col in collections]
        result["collections"] = names
        result["count"] = len(names)
        result["missing_expected"] = [name for name in expected if name not in names]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _safe_key_info(value):
    if not value:
        return {
            "present": False,
            "length": 0,
            "starts_with": None,
            "ends_with": None,
            "has_quotes": False,
            "has_whitespace": False,
        }

    text = str(value)
    stripped = text.strip()
    return {
        "present": bool(stripped),
        "length": len(stripped),
        "starts_with": stripped[:8],
        "ends_with": stripped[-4:],
        "has_quotes": stripped.startswith(("'", '"')) or stripped.endswith(("'", '"')),
        "has_whitespace": text != stripped,
    }


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
    chroma_health = _chroma_health()
    return jsonify({
        "status": "ok" if not notebook_error else "notebook_error",
        "notebook_error": notebook_error,
        "notebook_warnings": notebook_warnings,
        "notebook_loaded": bool(global_scope),
        "db_available": os.path.exists(DB_PATH),
        "has_answer": "answer" in global_scope,
        "has_rag_student_answer": "rag_student_answer" in global_scope,
        "has_protos": "protos" in global_scope,
        "has_chroma": chroma_health["available"],
        "chroma_collection_count": chroma_health["count"],
        "chroma_expected_count": chroma_health["expected_count"],
        "chroma_collections": chroma_health["collections"],
        "chroma_expected_collections": chroma_health["expected_collections"],
        "chroma_missing_expected": chroma_health["missing_expected"],
        "chroma_error": chroma_health["error"],
        "openai_key": {
            "environment": _safe_key_info(os.environ.get("OPENAI_API_KEY")),
            "notebook_value": _safe_key_info(global_scope.get("openai_api_key_value")),
            "client_loaded": global_scope.get("client") is not None,
        },
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
        detail = f"\n\nتفاصيل الخطأ: {notebook_error}" if notebook_error else ""
        return jsonify({"reply": "لم يتم تحميل نموذج المحادثة بعد. تحقق من سجلات Render لمعرفة سبب تعطل notebook." + detail}), 503

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
        detail = f"\n\nتفاصيل الخطأ: {notebook_error}" if notebook_error else ""
        return jsonify({"reply": "لم يتم تحميل نموذج المرشد بعد. تحقق من سجلات Render لمعرفة سبب تعطل notebook." + detail}), 503

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


@app.route("/debug-rag")
def debug_rag():
    import inspect

    q = request.args.get("q", "").strip()
    protos = global_scope.get("protos")

    out = {
        "query": q,
        "notebook_error": notebook_error,
        "has_chroma": "chroma" in global_scope,
        "has_protos": "protos" in global_scope,
        "has_answer": "answer" in global_scope,
        "has_retrieve_final": "retrieve_final" in global_scope,
        "has_retrieve_with_router_as_hint": "retrieve_with_router_as_hint" in global_scope,
        "has_rag_student_answer": "rag_student_answer" in global_scope,
        "signatures": {},
        "retrieve_final_attempts": [],
        "answer_debug": None,
        "rag_student_answer_debug": None,
        "error": None,
    }

    def safe_repr(x, limit=5000):
        try:
            return repr(x)[:limit]
        except Exception as e:
            return f"<repr failed: {type(e).__name__}: {e}>"

    def add_signature(name):
        fn = global_scope.get(name)
        if fn is None:
            out["signatures"][name] = None
            return
        try:
            out["signatures"][name] = str(inspect.signature(fn))
        except Exception as e:
            out["signatures"][name] = f"signature_error: {type(e).__name__}: {e}"

    for name in [
        "retrieve_final",
        "retrieve_with_router_as_hint",
        "answer",
        "rag_student_answer",
        "context_answers_question",
    ]:
        add_signature(name)

    try:
        retrieve_final = global_scope.get("retrieve_final")

        if retrieve_final is not None:
            attempts = [
                ("retrieve_final(q)", lambda: retrieve_final(q)),
                ("retrieve_final(query=q)", lambda: retrieve_final(query=q)),
                ("retrieve_final(q, protos)", lambda: retrieve_final(q, protos)),
                ("retrieve_final(q, protos=protos)", lambda: retrieve_final(q, protos=protos)),
                ("retrieve_final(q, protos=protos, mode='router')", lambda: retrieve_final(q, protos=protos, mode="router")),
            ]

            for label, call in attempts:
                try:
                    result = call()
                    out["retrieve_final_attempts"].append({
                        "call": label,
                        "ok": True,
                        "type": type(result).__name__,
                        "repr": safe_repr(result, 4000),
                    })
                    break
                except Exception as e:
                    out["retrieve_final_attempts"].append({
                        "call": label,
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                    })

        if "answer" in global_scope:
            try:
                a = global_scope["answer"](
                    query=q,
                    protos=protos,
                    mode="router",
                    return_debug=True,
                    use_web=True,
                )
                out["answer_debug"] = {
                    "ok": True,
                    "type": type(a).__name__,
                    "repr": safe_repr(a, 8000),
                }
            except Exception as e:
                out["answer_debug"] = {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }

        if "rag_student_answer" in global_scope:
            try:
                r = global_scope["rag_student_answer"](
                    question=q,
                    protos=protos,
                    retrieval_mode="router",
                )
                out["rag_student_answer_debug"] = {
                    "ok": True,
                    "type": type(r).__name__,
                    "repr": safe_repr(r, 8000),
                }
            except Exception as e:
                out["rag_student_answer_debug"] = {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }

        return jsonify(out)

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        return jsonify(out), 500
    
@app.route("/debug-chroma-query")
def debug_chroma_query():
    q = request.args.get("q", "").strip()
    collection_name = request.args.get("collection", "").strip()

    out = {
        "query": q,
        "collection": collection_name,
        "has_chroma": "chroma" in global_scope,
        "has_embed_query": "embed_query" in global_scope,
        "collection_exists": False,
        "collection_count": None,
        "raw_query": None,
        "error": None,
    }

    try:
        if not q:
            out["error"] = "Missing q"
            return jsonify(out), 400

        chroma = global_scope.get("chroma")
        if chroma is None:
            out["error"] = "No chroma client in global_scope"
            return jsonify(out), 500

        if not collection_name:
            collection_name = "faculty_offices"
            out["collection"] = collection_name

        col = chroma.get_collection(collection_name)
        out["collection_exists"] = True
        out["collection_count"] = col.count()

        # Use the same embedding path as the notebook if available.
        if "embed_query" in global_scope:
            qvec = global_scope["embed_query"](q)
            res = col.query(
                query_embeddings=[qvec],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
        else:
            res = col.query(
                query_texts=[q],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )

        out["raw_query"] = {
            "documents": res.get("documents"),
            "metadatas": res.get("metadatas"),
            "distances": res.get("distances"),
        }

        return jsonify(out)

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        return jsonify(out), 500
    
    
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
