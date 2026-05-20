from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import ast
import nbformat
import sqlite3
import os
import re
import shutil
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
    db_path = Path(db_dir) / "chroma.sqlite3"
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    try:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(collections)").fetchall()
        }
        if "schema_str" in columns or "config_json_str" in columns:
            return
        if "topic" not in columns:
            conn.execute("ALTER TABLE collections ADD COLUMN topic TEXT")
            conn.commit()
            print("[PATCH] Added missing Chroma collections.topic column.")
    finally:
        conn.close()


class _CollectionRef:
    def __init__(self, name):
        self.name = name


def _safe_chroma_list_collections(client):
    try:
        collections = client.list_collections()
        return [
            col if hasattr(col, "name") else _CollectionRef(str(col))
            for col in collections
        ]
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


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return "\n".join(str(item) for item in value if item is not None)
    return str(value)


def _as_source_list(value):
    if value is None:
        return []
    if isinstance(value, dict):
        merged = []
        for items in value.values():
            merged.extend(_as_source_list(items))
        return merged
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, (tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _compact_sources(value, limit=12):
    return [source[:500] for source in _as_source_list(value)[:limit]]


def _student_data_question(question):
    q = _as_text(question).strip().lower()
    if re.search(r"\b\d{6,}\b", q):
        return True

    student_markers = [
        "بيانات الطالب",
        "بيانات الطالبة",
        "سجل الطالب",
        "سجل الطالبة",
        "معدل",
        "gpa",
        "حضور",
        "غياب",
        "درجات",
        "متعثر",
        "متعثره",
        "متعثرة",
        "انذار",
        "إنذار",
        "درسها",
        "موادها",
    ]
    return any(marker in q for marker in student_markers)


def _summarize_retrieval_result(result, include_preview=False):
    if not isinstance(result, dict):
        return {
            "ok": False,
            "error": f"retrieve_final returned {type(result).__name__}",
            "context_length": 0,
            "sources": [],
        }

    context = _as_text(result.get("context_text", ""))
    summary = {
        "ok": bool(context.strip()),
        "error": None,
        "mode": result.get("mode"),
        "variant": result.get("variant"),
        "route_mode": result.get("route_mode"),
        "route_candidates": _compact_sources(result.get("route_candidates"), limit=8),
        "retrieval_scope": result.get("retrieval_scope"),
        "context_length": len(context),
        "sources": _compact_sources(result.get("sources")),
        "source_count": len(_as_source_list(result.get("sources"))),
    }
    if include_preview:
        summary["context_preview"] = context[:1600]
    return summary


def _probe_internal_retrieval(question, mode="all", include_preview=False):
    fn = global_scope.get("retrieve_final")
    if fn is None:
        return {
            "ok": False,
            "error": "retrieve_final is not loaded",
            "context_length": 0,
            "sources": [],
        }

    try:
        result = fn(query=question, protos=global_scope.get("protos"), mode=mode)
        return _summarize_retrieval_result(result, include_preview=include_preview)
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "context_length": 0,
            "sources": [],
        }


_AR_SEARCH_STOPWORDS = {
    "ما", "ماهي", "ماهي", "ما", "هي", "هو", "متى", "متي", "كيف", "كم", "من",
    "في", "عن", "على", "الى", "إلى", "هل", "الطالب", "الطالبة",
    "يقدر", "ياخذ", "تكون", "يكون", "اقدر", "أقدر", "وش", "ايش",
}


def _normalize_ar_search(text):
    text = _as_text(text).lower()
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = re.sub(r"[^\w\s\u0600-\u06FF-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _search_terms(question):
    normalized = _normalize_ar_search(question)
    terms = []
    for term in normalized.split():
        if len(term) < 3 or term in _AR_SEARCH_STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)

    aliases = {
        "صيفي": ["الصيفي", "صيف"],
        "تدريب": ["التدريب", "تعاوني", "التعاوني", "coop"],
        "ذكاء": ["الذكاء"],
        "اصطناعي": ["الاصطناعي"],
        "متطلبات": ["متطلب", "المتطلبات", "شروط"],
    }
    for term in list(terms):
        for alias in aliases.get(term, []):
            if alias not in terms:
                terms.append(alias)

    if any(term in normalized for term in ["متي", "متى", "يقدر", "ياخذ"]) and any(
        term in normalized for term in ["تدريب", "صيفي", "تعاوني"]
    ):
        for alias in ["يشترط", "للتسجيل", "التسجيل", "وحدة", "وحده", "120"]:
            if alias not in terms:
                terms.append(alias)

    return terms[:14]


def _vector_db_path():
    chroma_dir = global_scope.get("CHROMA_DIR") or "data/chroma_uja1"
    db_path = BASE_DIR / chroma_dir / "chroma.sqlite3"
    if db_path.exists():
        return db_path

    fallback = BASE_DIR / "data" / "chroma_uja1" / "chroma.sqlite3"
    return fallback if fallback.exists() else None


def _load_vector_text_rows():
    db_path = _vector_db_path()
    if not db_path:
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in conn.execute("""
                SELECT
                    e.id AS row_id,
                    f.string_value AS document,
                    c.name AS collection,
                    MAX(CASE WHEN m.key = 'source_ref' THEN m.string_value END) AS source_ref,
                    MAX(CASE WHEN m.key = 'parent_id' THEN m.string_value END) AS parent_id,
                    MAX(CASE WHEN m.key = 'doc_type' THEN m.string_value END) AS doc_type
                FROM embedding_fulltext_search f
                JOIN embeddings e ON e.id = f.rowid
                JOIN segments s ON s.id = e.segment_id
                JOIN collections c ON c.id = s.collection
                LEFT JOIN embedding_metadata m ON m.id = e.id
                GROUP BY e.id, f.string_value, c.name
            """)
        ]
    except Exception as e:
        print(f"[DIRECT VECTOR] Failed reading Chroma SQLite text rows: {e}")
        return []
    finally:
        conn.close()


def _score_vector_row(row, terms, question):
    text = _normalize_ar_search(row.get("document", ""))
    if not text:
        return 0

    score = 0
    for term in terms:
        if term in text:
            score += 3
        if text.count(term) > 1:
            score += min(text.count(term), 4)

    q = _normalize_ar_search(question)
    collection = row.get("collection", "")
    if any(t in q for t in ["ذكاء", "اصطناعي", "تخصص", "متطلبات"]):
        if collection in {"degree_plans", "specialization"}:
            score += 5
    if any(t in q for t in ["تدريب", "صيفي", "تعاوني"]):
        if collection in {"coop_rules", "coop_replies", "academic_calendar", "regulations"}:
            score += 5

    return score


def _relevant_excerpt(text, terms, max_chars=1600):
    text = _as_text(text).strip()
    if len(text) <= max_chars:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    picked = []
    for line in lines:
        norm = _normalize_ar_search(line)
        if any(term in norm for term in terms):
            picked.append(line)
        if len("\n".join(picked)) >= max_chars:
            break

    excerpt = "\n".join(picked).strip() or text[:max_chars]
    return excerpt[:max_chars].strip()


def _direct_vector_answer(question):
    terms = _search_terms(question)
    if not terms:
        return None

    rows = _load_vector_text_rows()
    scored = []
    for row in rows:
        score = _score_vector_row(row, terms, question)
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda item: (-item[0], item[1].get("collection", ""), item[1].get("row_id", 0)))
    top = scored[:4]
    if not top:
        return None

    blocks = []
    sources = []
    for score, row in top:
        source = row.get("source_ref") or f"{row.get('collection')}:{row.get('row_id')}"
        sources.append(source)
        excerpt = _relevant_excerpt(row.get("document", ""), terms)
        blocks.append(f"[{source}] ({row.get('collection')})\n{excerpt}")

    context = "\n\n".join(blocks).strip()
    answer = (
        "اعتمادًا على قاعدة المعرفة الداخلية:\n\n"
        + context[:2600]
        + "\n\nالمصادر الداخلية: "
        + "، ".join(_compact_sources(sources, limit=5))
    )

    return {
        "answer": answer,
        "sources": {"kb": sources, "uj_web": [], "full_web": []},
        "contexts": {"kb": context, "uj_web": "", "full_web": ""},
        "trace": {
            "kb_answered": True,
            "retrieval_method": "direct_chroma_sqlite_text_search",
            "terms": terms,
            "hits": len(scored),
            "web_skipped_reason": "internal_vector_context_found",
        },
    }


def _quick_grounded_answer(question, context_kb, sources_kb):
    context_kb = _as_text(context_kb).strip()
    sources_kb = _compact_sources(sources_kb, limit=8)
    if not context_kb:
        return ""

    context_for_model = context_kb[:6000]
    client = global_scope.get("client")
    model = global_scope.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "أجب بالعربية فقط اعتمادًا على النص الداخلي المرفق. "
                            "لا تستخدم الويب ولا تضف معلومات غير موجودة في النص. "
                            "إذا كان النص لا يحتوي الإجابة، قل ذلك بوضوح."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"السؤال:\n{question}\n\n"
                            f"النص الداخلي من قاعدة المتجهات:\n{context_for_model}\n\n"
                            f"المصادر:\n{sources_kb}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=260,
                timeout=25,
            )
            answer = (resp.choices[0].message.content or "").strip()
            if answer:
                return answer
        except Exception as e:
            print(f"[VECTOR FIRST] Quick grounded LLM failed; returning vector excerpt: {e}")

    excerpt = context_kb[:1800].strip()
    return (
        "وجدت معلومات مرتبطة في قاعدة المعرفة الداخلية، لكن تعذر صياغتها بالنموذج الآن. "
        "أقرب نص مسترجع:\n\n"
        + excerpt
    )


def _internal_vector_answer(question, protos):
    direct = _direct_vector_answer(question)
    if direct is not None:
        return direct

    retrieval_fn = global_scope.get("retrieve_with_router_as_hint") or global_scope.get("retrieve_final")

    if retrieval_fn is not None:
        try:
            if "retrieve_with_router_as_hint" in global_scope:
                retrieval_result = retrieval_fn(
                    query=question,
                    protos=protos,
                    mode="router",
                    route_decision=None,
                    use_bm25=True,
                )
            else:
                retrieval_result = retrieval_fn(
                    query=question,
                    protos=protos,
                    mode="router",
                    use_bm25=True,
                )

            context_kb = _as_text(retrieval_result.get("context_text", "")) if isinstance(retrieval_result, dict) else ""
            sources_kb = retrieval_result.get("sources", []) if isinstance(retrieval_result, dict) else []

            if context_kb.strip() and _as_source_list(sources_kb):
                answer_text = _quick_grounded_answer(question, context_kb, sources_kb)

                return {
                    "answer": _as_text(answer_text),
                    "sources": {"kb": sources_kb, "uj_web": [], "full_web": []},
                    "contexts": {"kb": context_kb, "uj_web": "", "full_web": ""},
                    "trace": {
                        "kb_answered": True,
                        "uj_web_answered": False,
                        "full_web_answered": False,
                        "web_skipped_reason": "internal_vector_context_found",
                    },
                    "retrieval": retrieval_result,
                }
        except Exception as e:
            print(f"[VECTOR FIRST] Internal retrieval failed, trying web fallback flow: {e}")

    answer_fn = global_scope.get("answer")
    if answer_fn is None:
        return {"answer": "لم يتم تحميل نموذج الإجابة من قاعدة المعرفة الداخلية بعد."}

    result = answer_fn(
        query=question,
        protos=protos,
        mode="all",
        return_debug=True,
        use_deep=False,
        use_web=True,
    )

    if not isinstance(result, dict):
        return {"answer": _as_text(result)}

    sources = result.get("sources", {}) or {}
    contexts = result.get("contexts", {}) or {}
    trace = result.get("trace", {}) or {}

    kb_sources = sources.get("kb", []) if isinstance(sources, dict) else sources
    uj_sources = sources.get("uj_web", []) if isinstance(sources, dict) else []
    web_sources = sources.get("full_web", []) if isinstance(sources, dict) else []
    kb_context = contexts.get("kb", "") if isinstance(contexts, dict) else ""
    uj_context = contexts.get("uj_web", "") if isinstance(contexts, dict) else ""
    web_context = contexts.get("full_web", "") if isinstance(contexts, dict) else ""

    has_internal_context = (
        bool(_as_source_list(kb_sources))
        or len(_as_text(kb_context).strip()) >= 120
        or bool(isinstance(trace, dict) and trace.get("kb_answered"))
    )

    has_web_context = (
        bool(_as_source_list(uj_sources))
        or bool(_as_source_list(web_sources))
        or len(_as_text(uj_context).strip()) >= 120
        or len(_as_text(web_context).strip()) >= 120
    )

    if has_internal_context or has_web_context:
        return result

    return {
        "answer": "لا توجد معلومات كافية في قاعدة المعرفة الداخلية أو مصادر الويب المتاحة للإجابة عن هذا السؤال بدقة.",
        "sources": {"kb": [], "uj_web": [], "full_web": []},
        "trace": trace,
    }


def _answer_flow_probe(question):
    answer_fn = global_scope.get("answer")
    if answer_fn is None:
        return {"ok": False, "error": "answer is not loaded"}

    try:
        result = answer_fn(
            query=question,
            protos=global_scope.get("protos"),
            mode="all",
            return_debug=True,
            use_deep=True,
            use_web=True,
        )
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    if not isinstance(result, dict):
        return {
            "ok": bool(_as_text(result).strip()),
            "answer_preview": _as_text(result)[:700],
            "error": None,
        }

    contexts = result.get("contexts", {}) or {}
    sources = result.get("sources", {}) or {}
    trace = result.get("trace", {}) or {}

    return {
        "ok": True,
        "answer_preview": _as_text(result.get("answer", ""))[:700],
        "trace": trace,
        "context_lengths": {
            "kb": len(_as_text(contexts.get("kb", ""))) if isinstance(contexts, dict) else 0,
            "uj_web": len(_as_text(contexts.get("uj_web", ""))) if isinstance(contexts, dict) else 0,
            "full_web": len(_as_text(contexts.get("full_web", ""))) if isinstance(contexts, dict) else 0,
        },
        "source_counts": {
            "kb": len(_as_source_list(sources.get("kb", []))) if isinstance(sources, dict) else 0,
            "uj_web": len(_as_source_list(sources.get("uj_web", []))) if isinstance(sources, dict) else 0,
            "full_web": len(_as_source_list(sources.get("full_web", []))) if isinstance(sources, dict) else 0,
        },
        "sources": {
            "kb": _compact_sources(sources.get("kb", [])) if isinstance(sources, dict) else [],
            "uj_web": _compact_sources(sources.get("uj_web", [])) if isinstance(sources, dict) else [],
            "full_web": _compact_sources(sources.get("full_web", [])) if isinstance(sources, dict) else [],
        },
        "error": None,
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
    payload = {
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
    }

    if request.args.get("probe", "").lower() in {"1", "true", "yes"}:
        q = request.args.get("q", "ما متطلبات تخصص الذكاء الاصطناعي؟").strip()
        payload["retrieval_probe"] = _probe_internal_retrieval(
            question=q,
            mode=request.args.get("mode", "all").strip() or "all",
            include_preview=True,
        )

    return jsonify(payload)


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
@app.route("/debug-rag")
def debug_rag():
    question = request.args.get("q", "ما متطلبات تخصص الذكاء الاصطناعي؟").strip()
    mode = request.args.get("mode", "all").strip() or "all"
    payload = {
        "question": question,
        "mode": mode,
        "retrieval": _probe_internal_retrieval(
            question=question,
            mode=mode,
            include_preview=True,
        ),
        "chroma": _chroma_health(),
        "openai_key": {
            "environment": _safe_key_info(os.environ.get("OPENAI_API_KEY")),
            "notebook_value": _safe_key_info(global_scope.get("openai_api_key_value")),
            "client_loaded": global_scope.get("client") is not None,
        },
    }

    if request.args.get("answer", "").lower() in {"1", "true", "yes"}:
        payload["answer_flow"] = _answer_flow_probe(question)
    else:
        payload["answer_flow"] = "skipped; add &answer=1 to run the slower full answer flow"

    return jsonify(payload)


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

    use_student_path = _student_data_question(question)

    if use_student_path and "rag_student_answer" not in global_scope:
        detail = f"\n\nتفاصيل الخطأ: {notebook_error}" if notebook_error else ""
        return jsonify({"reply": "لم يتم تحميل نموذج المرشد بعد. تحقق من سجلات Render لمعرفة سبب تعطل notebook." + detail}), 503

    if not use_student_path and "answer" not in global_scope:
        detail = f"\n\nتفاصيل الخطأ: {notebook_error}" if notebook_error else ""
        return jsonify({"reply": "لم يتم تحميل قاعدة المعرفة الداخلية بعد. تحقق من سجلات Render لمعرفة سبب تعطل notebook." + detail}), 503

    try:
        protos = global_scope.get("protos", None)

        if use_student_path:
            result = global_scope["rag_student_answer"](
                question=question,
                protos=protos,
                retrieval_mode="router"
            )
        else:
            result = _internal_vector_answer(question, protos)

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
