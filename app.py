diff --git a/app.py b/app.py
index eee452400664bb40cbadc4fed7853686b84f2536..25d706f2b7b8475300133ff2b7fdc87e6e135e09 100644
--- a/app.py
+++ b/app.py
@@ -1,136 +1,227 @@
 from flask import Flask, request, jsonify, render_template
 from flask_cors import CORS
+import ast
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
+notebook_warnings = []
 
 # Render/gunicorn can start the app from a different current working directory.
 # The notebook contains many relative data paths, so anchor execution at the repo.
 os.chdir(BASE_DIR)
 
 NOTEBOOK_PATH_REPLACEMENTS = {
     '"data/vector_database_updated.zip"': '"data/vector_database_updated (1).zip"',
     "chroma.list_collections()": "_safe_chroma_list_collections(chroma)",
 }
 
+NOTEBOOK_SKIP_PREFIXES = ("!", "%")
+DEMO_ASSIGN_NAMES = {"test_questions", "test_questions1"}
+DEMO_CALL_NAMES = {"run_system_test"}
+
+
+def _clean_notebook_cell_source(cell_source: str):
+    if not cell_source or not cell_source.strip():
+        return "", "empty"
+    lines = [ln.rstrip() for ln in cell_source.splitlines()]
+    cleaned = [ln for ln in lines if ln.strip() and not ln.strip().startswith(NOTEBOOK_SKIP_PREFIXES)]
+    if not cleaned:
+        return "", "shell/magic-only"
+    return "\n".join(cleaned).strip(), ""
+
+
+def _is_demo_assign(stmt: ast.stmt) -> bool:
+    if not isinstance(stmt, ast.Assign):
+        return False
+    target_names = {t.id for t in stmt.targets if isinstance(t, ast.Name)}
+    return bool(target_names) and target_names.issubset(DEMO_ASSIGN_NAMES)
+
+
+def _is_demo_call(stmt: ast.stmt) -> bool:
+    return (
+        isinstance(stmt, ast.Expr)
+        and isinstance(stmt.value, ast.Call)
+        and isinstance(stmt.value.func, ast.Name)
+        and stmt.value.func.id in DEMO_CALL_NAMES
+    )
+
+
+def _is_bare_expr(stmt: ast.stmt) -> bool:
+    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Name)
+
+
+def _prepare_notebook_cell_code(cell_source: str):
+    """Return (code_obj|None, skip_reason)."""
+    cleaned_code, cleaned_reason = _clean_notebook_cell_source(cell_source)
+    if cleaned_reason:
+        return None, cleaned_reason
+
+    try:
+        tree = ast.parse(cleaned_code, mode="exec")
+    except SyntaxError:
+        return compile(cleaned_code, "<notebook-cell>", "exec"), ""
+
+    body = list(tree.body)
+    if not body:
+        return None, "empty"
+
+    kept = []
+    demo_removed = 0
+    for stmt in body:
+        if _is_demo_assign(stmt) or _is_demo_call(stmt) or _is_bare_expr(stmt):
+            demo_removed += 1
+            continue
+        kept.append(stmt)
+
+    if not kept:
+        if len(body) == 1 and _is_bare_expr(body[0]):
+            return None, "bare expression"
+        if all(_is_demo_assign(s) for s in body):
+            return None, "demo variable assignment"
+        if all(_is_demo_call(s) for s in body):
+            return None, "demo function call"
+        return None, "non-production-only cell"
+
+    new_tree = ast.Module(body=kept, type_ignores=[])
+    ast.fix_missing_locations(new_tree)
+    reason = "demo statements removed" if demo_removed else ""
+    return compile(new_tree, "<notebook-cell>", "exec"), reason
+
 
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
-    global global_scope, notebook_error
+    global global_scope, notebook_error, notebook_warnings
     notebook_error = None
+    notebook_warnings = []
 
     if not NOTEBOOK_PATH.exists():
         notebook_error = f"Notebook file not found: {NOTEBOOK_PATH}"
         print(notebook_error)
         global_scope = {}
         return
 
     with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
         nb = nbformat.read(f, as_version=4)
 
     global_scope = {
         "_safe_chroma_list_collections": _safe_chroma_list_collections,
     }
 
     for idx, cell in enumerate(nb.cells):
         if cell.cell_type == "code":
             try:
-                lines = cell.source.splitlines()
-                cleaned_lines = []
-                for line in lines:
-                    stripped = line.strip()
-                    if stripped.startswith("!") or stripped.startswith("%"):
-                        continue
-                    cleaned_lines.append(line)
-                cleaned_code = "\n".join(cleaned_lines).strip()
+                cleaned_code, _ = _clean_notebook_cell_source(cell.source)
                 if not cleaned_code:
+                    notebook_warnings.append(f"Cell {idx} skipped: shell/magic-only or empty")
                     continue
                 for old, new in NOTEBOOK_PATH_REPLACEMENTS.items():
                     cleaned_code = cleaned_code.replace(old, new)
-                exec(cleaned_code, global_scope)
+
+                code_obj, reason = _prepare_notebook_cell_code(cleaned_code)
+                if code_obj is None:
+                    notebook_warnings.append(f"Cell {idx} skipped: {reason}")
+                    continue
+                if reason:
+                    notebook_warnings.append(f"Cell {idx}: {reason}")
+
+                exec(code_obj, global_scope)
             except Exception as e:
-                notebook_error = f"Notebook failed while loading cell {idx}: {e}"
+                warning = f"Cell {idx} failed: {type(e).__name__}: {e}"
+                notebook_warnings.append(warning)
                 print("=" * 80)
-                print(f"[NOTEBOOK LOAD ERROR] Cell index: {idx}")
-                print(f"[NOTEBOOK LOAD ERROR] Exception: {type(e).__name__}: {e}")
+                print(f"[NOTEBOOK LOAD WARNING] Cell index: {idx}")
+                print(f"[NOTEBOOK LOAD WARNING] Exception: {type(e).__name__}: {e}")
                 print("[CELL SOURCE START]")
                 print(cell.source[:3000])
                 print("[CELL SOURCE END]")
                 traceback.print_exc()
+                print("[NOTEBOOK LOAD WARNING] Continuing to load remaining cells.")
                 print("=" * 80)
-                global_scope = {}
-                return
+                continue
+
+    required = ["answer", "rag_student_answer", "protos", "chroma"]
+    missing_required = [name for name in required if name not in global_scope]
+    if missing_required:
+        notebook_error = (
+            "Notebook loaded with errors and critical handlers are missing: "
+            + ", ".join(missing_required)
+            + ". "
+            + (" | ".join(notebook_warnings[:3]) if notebook_warnings else "")
+        )
+        print(f"[NOTEBOOK LOAD ERROR] {notebook_error}")
+        global_scope = {}
+        return
 
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
@@ -384,134 +475,198 @@ def _chroma_health():
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
 
 
+def _retrieval_probe(question="ما هي شروط التدريب التعاوني؟"):
+    fn = global_scope.get("retrieve_final")
+    protos = global_scope.get("protos")
+    if fn is None:
+        return {"ok": False, "error": "retrieve_final not loaded", "context_len": 0, "sources": []}
+    try:
+        result = fn(query=question, protos=protos, mode="router")
+        if not isinstance(result, dict):
+            return {"ok": False, "error": "retrieve_final returned non-dict", "context_len": 0, "sources": []}
+        context = result.get("context_text", "") or ""
+        return {
+            "ok": len(context.strip()) > 0,
+            "context_len": len(context),
+            "sources": result.get("sources", []) or [],
+            "error": None,
+        }
+    except Exception as e:
+        return {"ok": False, "error": f"{type(e).__name__}: {e}", "context_len": 0, "sources": []}
+
+
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
+    probe = _retrieval_probe()
     return jsonify({
         "status": "ok" if not notebook_error else "notebook_error",
         "notebook_error": notebook_error,
         "notebook_loaded": bool(global_scope),
+        "notebook_warnings": notebook_warnings,
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
+        "retrieval_probe": probe,
+        "retrieval_probe_ok": probe.get("ok", False),
+        "retrieval_probe_context_len": probe.get("context_len", 0),
+        "retrieval_probe_error": probe.get("error"),
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
+@app.route("/debug-rag")
+def debug_rag():
+    q = (request.args.get("q") or "ما هي شروط التدريب التعاوني؟").strip()
+    fn = global_scope.get("retrieve_final")
+    protos = global_scope.get("protos")
+
+    if fn is None:
+        return jsonify({"error": "retrieve_final not loaded"}), 503
+
+    try:
+        result = fn(query=q, protos=protos, mode="router")
+        if not isinstance(result, dict):
+            return jsonify({"error": "retrieve_final returned non-dict", "type": str(type(result))}), 500
+        context = result.get("context_text", "") or ""
+        return jsonify({
+            "question": q,
+            "route_candidates": {
+                "kb": result.get("kb_route"),
+                "uj": result.get("uj_route"),
+                "web": result.get("web_route"),
+                "routing": result.get("routing"),
+            },
+            "sources": result.get("sources", []) or [],
+            "context_length": len(context),
+            "context_preview": context[:1000],
+            "retrieval_error": None,
+        })
+    except Exception as e:
+        return jsonify({
+            "question": q,
+            "route_candidates": {},
+            "sources": [],
+            "context_length": 0,
+            "context_preview": "",
+            "retrieval_error": f"{type(e).__name__}: {e}",
+        }), 500
+
+
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
 
