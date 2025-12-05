# ============================
# smartbvb.py  (PART 1 / 3)
# ============================
"""
FastAPI backend (smartbvb) — clean production/testing-ready single-file API.

University use-case:

- Admin / Departments / Semesters / Stores
- Each Sem stores a Gemini API key (used for all operations under that sem)
- Upload / delete files (PDF / images / DOCX) → OCR → text → index into Gemini File Search
- ASK endpoint:
    - Extract system_prompt + user_query from the student's message
    - Select relevant stores via Gemini
    - Run RAG per-store (File Search)
    - Merge answers, applying the system_prompt only at final merge

Important constraints:

- NO session management
- DO NOT store question/answer text anywhere
  (only metadata about stores/files is kept)
- CORS enabled for testing: allows calls from anywhere
"""

from PIL import Image  # OCR image handling
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from docx import Document

import os
import time
import json
import shutil
import re
import requests
import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import mimetypes  # MIME detection
import tempfile   # temp files for OCR etc.

# OCR-related imports
import pytesseract
import fitz  # PyMuPDF
import docx2txt

logger = logging.getLogger("smartbvb")
logging.basicConfig(level=logging.INFO)

# ---------------- Defensive imports ----------------
try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import aiofiles

    FASTAPI_AVAILABLE = True
except Exception as e:
    logger.warning("FastAPI or aiofiles import failed: %s. Running in limited mode.", e)
    FASTAPI_AVAILABLE = False

    class HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: Optional[str] = None):
            super().__init__(f"HTTPException {status_code}: {detail}")
            self.status_code = status_code
            self.detail = detail

    class JSONResponse:
        def __init__(self, content, status_code: int = 200):
            self.content = content
            self.status_code = status_code

        def __repr__(self):
            return f"JSONResponse(status_code={self.status_code}, content={self.content})"

    def UploadFile(*args, **kwargs):
        class _UF:
            pass
        return _UF

    def File(*args, **kwargs):
        return None

    def Form(*args, **kwargs):
        return None

    class aiofiles:
        class open:
            def __init__(self, path, mode="rb"):
                self._path = path
                self._mode = mode
                self._f = None

            async def __aenter__(self):
                self._f = open(self._path, self._mode)
                return self._f

            async def __aexit__(self, exc_type, exc, tb):
                try:
                    if self._f and not self._f.closed:
                        self._f.close()
                except Exception:
                    pass

# Try to import google-genai SDK; if missing, set to None and endpoints will return clear error
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# ---------------- CONFIG ----------------

# Path for persistent JSON metadata file (can be overridden with env var)
# Use /tmp for Cloud Run (only writable dir)
DEFAULT_DATA_ROOT = os.getenv("DATA_ROOT", "/tmp")

DATA_FILE = os.getenv("DATA_FILE", str(Path(DEFAULT_DATA_ROOT) / "gemini_stores.json"))
UPLOAD_ROOT = Path(os.getenv("UPLOAD_ROOT", str(Path(DEFAULT_DATA_ROOT) / "uploads")))

MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", 50 * 1024 * 1024))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 2))
GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Simple admin creds for local/dev testing (not secure for production)
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "mrsadiq471@gmail.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "sadiq^8064")

# ----------------------------------------
app = FastAPI()

# CORS — allow everything for testing (you asked calls can come from anywhere)
try:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass


# ---------------- Helpers: persistence ----------------

def ensure_dirs():
    try:
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.debug("Could not create upload root: %s", e)


def _ensure_data_file_initial():
    """
    Ensure a usable DATA_FILE exists. If DATA_FILE points to an existing directory,
    we will write a gemini_stores.json file inside that directory and adjust DATA_FILE.

    Returns the path used for the data file (may update global DATA_FILE).
    """
    global DATA_FILE
    base = {
        "file_stores": {},
        "departments": {}
    }

    # If DATA_FILE is a directory, recover by using DATA_FILE/gemini_stores.json
    if os.path.isdir(DATA_FILE):
        logger.warning(
            "DATA_FILE path %s is a directory. Using %s/gemini_stores.json instead.",
            DATA_FILE, DATA_FILE
        )
        alt = os.path.join(DATA_FILE, "gemini_stores.json")
        try:
            os.makedirs(DATA_FILE, exist_ok=True)
            if not os.path.exists(alt):
                with open(alt, "w") as f:
                    json.dump(base, f, indent=2)
            DATA_FILE = alt
            return DATA_FILE
        except Exception as e:
            logger.error("Failed to initialize alternative data file %s: %s", alt, e)
            raise

    # Ensure parent directory exists
    parent = os.path.dirname(DATA_FILE) or "/"
    try:
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
    except Exception as e:
        logger.error(
            "Could not create parent directory for DATA_FILE (%s): %s",
            parent, e
        )
        raise

    # If path somehow became a directory, error (can't proceed)
    if os.path.isdir(DATA_FILE):
        raise RuntimeError(
            f"DATA_FILE path {DATA_FILE} is a directory and cannot be used as a file."
        )

    # Create the file if missing
    if not os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "w") as f:
                json.dump(base, f, indent=2)
        except Exception as e:
            logger.error("Failed to create DATA_FILE %s: %s", DATA_FILE, e)
            raise

    return DATA_FILE


def load_data() -> Dict[str, Any]:
    """
    Load JSON metadata from DATA_FILE. If file is missing/corrupt, attempt to reinitialize.
    """
    # If DATA_FILE is a directory, try to recover by creating a file inside it
    if os.path.isdir(DATA_FILE):
        _ensure_data_file_initial()

    if not os.path.exists(DATA_FILE):
        _ensure_data_file_initial()

    if not os.path.isfile(DATA_FILE):
        raise RuntimeError(f"Data file path is not a regular file: {DATA_FILE}")

    with open(DATA_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            logger.warning("DATA_FILE %s corrupted or empty. Reinitializing.", DATA_FILE)
            _ensure_data_file_initial()
            with open(DATA_FILE, "r") as f2:
                return json.load(f2)


def save_data(data: Dict[str, Any]):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


# Run initial ensures
ensure_dirs()
_ensure_data_file_initial()


# ---------------- Utilities ----------------

def clean_filename(name: str, max_len: int = 180) -> str:
    if not name:
        return "file"
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"^\.+", "", name)
    name = re.sub(r"[^A-Za-z0-9_\-\.]", "_", name)
    name = re.sub(r"__+", "_", name)
    name = re.sub(r"\.\.+", ".", name)
    if len(name) > max_len:
        name = name[:max_len]
    if not name:
        return "file"
    return name


def guess_mime_type(filename: str, fallback: str = "application/octet-stream") -> str:
    """
    Robust MIME type detection.
    - Try Python's mimetypes
    - Fallback to generic octet-stream (Gemini will still try for many)
    """
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        return mime
    return fallback


def normalize_mime_type(mime: Optional[str]) -> Optional[str]:
    """
    Ensure MIME type is clean 'type/subtype'.
    - Remove parameters like '; charset=binary'
    - Validate correct structure
    """
    if not mime:
        return None

    mime = mime.strip()

    # Remove anything after ';'
    if ";" in mime:
        mime = mime.split(";", 1)[0].strip()

    # Must contain '/'
    if "/" not in mime:
        return None

    return mime


# ---------------- OCR HELPERS (PDF / IMAGE / DOCX → TEXT) ----------------

async def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Convert a PDF (bytes) into text using PyMuPDF + Tesseract.
    Good for scanned PDFs, question papers, notes, etc.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        pdf_path = tmp.name

    pdf = fitz.open(pdf_path)
    text_output = []

    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_tmp:
            img_tmp.write(img_bytes)
            img_path = img_tmp.name

        image = Image.open(img_path)
        extracted_text = pytesseract.image_to_string(image)
        text_output.append(f"--- PAGE {page_num + 1} ---\n{extracted_text}")

    return "\n".join(text_output)


async def extract_text_from_image(file_bytes: bytes) -> str:
    """
    Convert an image (bytes) into text using Tesseract.
    For photographed notes, board pictures, etc.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        image_path = tmp.name

    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


async def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Convert a DOCX (bytes) into plain text using docx2txt.
    For lesson plans, typed notes, etc.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        docx_path = tmp.name

    extracted_text = docx2txt.process(docx_path)
    return extracted_text


# ---------------- Gemini helpers (basic) ----------------

def init_gemini_client(api_key: str):
    if genai is None:
        raise RuntimeError("google-genai SDK is not installed on the server.")
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")


def wait_for_operation(client, operation):
    op = operation
    while not getattr(op, "done", False):
        time.sleep(POLL_INTERVAL)
        try:
            if hasattr(client, "operations") and hasattr(client.operations, "get"):
                op = client.operations.get(op)
        except Exception:
            pass

    if getattr(op, "error", None):
        raise RuntimeError(f"Operation failed: {op.error}")
    return op


def rest_list_documents_for_store(file_search_store_name: str, api_key: str):
    url = f"{GEMINI_REST_BASE}/{file_search_store_name}/documents"
    params = {"key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("documents", [])
    except Exception:
        return []


# ---------------- Admin endpoints ----------------

@app.post("/admin/login")
def admin_login(email: str = Form(...), password: str = Form(...)):
    """
    Simple admin login for development use. Returns a success message.
    (No session/token management — per requirement.)
    """
    if email != ADMIN_EMAIL or password != ADMIN_PASSWORD:
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)

    return {"success": True, "message": "Admin authenticated"}


@app.post("/admin/departments/create")
def create_department(department_name: str = Form(...)):
    data = load_data()

    if not department_name:
        raise HTTPException(status_code=400, detail="department_name required")
    departments = data.setdefault("departments", {})
    if department_name in departments:
        return JSONResponse({"error": "Department already exists"}, status_code=400)
    departments[department_name] = {"sems": {}}
    save_data(data)
    return {"success": True, "department": department_name}


@app.get("/admin/departments")
def list_departments():
    data = load_data()
    return {"success": True, "departments": list(data.get("departments", {}).keys())}


@app.delete("/admin/departments/{department}")
def delete_department(department: str):
    data = load_data()
    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")

    # iterate sems and delete
    sems = data["departments"][department].get("sems", {})
    for sem_name, sem_meta in list(sems.items()):
        stores = sem_meta.get("stores", [])
        sem_key = sem_meta.get("gemini_api_key")
        for store_name in list(stores):
            if store_name in data.get("file_stores", {}):
                fs = data["file_stores"][store_name].get("file_search_store_name")
                try:
                    if sem_key and genai is not None:
                        client = init_gemini_client(sem_key)
                        try:
                            client.file_search_stores.delete(
                                name=fs,
                                config={"force": True}
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    del data["file_stores"][store_name]
                except Exception:
                    pass
        folder = UPLOAD_ROOT / department / sem_name
        if folder.exists():
            try:
                shutil.rmtree(folder)
            except Exception:
                pass
    del data["departments"][department]
    save_data(data)
    return {"success": True, "deleted_department": department}


@app.post("/admin/departments/{department}/sems/create")
def create_sem(
    department: str,
    sem_name: str = Form(...),
    gemini_api_key: str = Form(...)
):
    data = load_data()

    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")

    sems = data["departments"][department].setdefault("sems", {})
    if sem_name in sems:
        return JSONResponse(
            {"error": "Sem already exists in the department"},
            status_code=400
        )

    # validate GEMINI API KEY using real call
    try:
        client = init_gemini_client(gemini_api_key)
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents="ping"
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"Invalid Gemini API key: {e}"},
            status_code=400
        )

    # if key valid → create sem
    sems[sem_name] = {
        "gemini_api_key": gemini_api_key,
        "stores": [],
        "total_size_bytes": 0
    }

    save_data(data)

    return {
        "success": True,
        "department": department,
        "sem": sem_name
    }


@app.get("/admin/departments/{department}/sems")
def list_sems(department: str):
    data = load_data()

    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")

    return {
        "success": True,
        "sems": list(data["departments"][department].get("sems", {}).keys())
    }
# ============================
# smartbvb.py  (PART 2 / 3)
# ============================

@app.delete("/admin/departments/{department}/sems/{sem}")
def delete_sem(department: str, sem: str):
    data = load_data()

    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")
    if sem not in data["departments"][department].get("sems", {}):
        raise HTTPException(status_code=404, detail="Sem not found")

    sem_meta = data["departments"][department]["sems"][sem]
    sem_key = sem_meta.get("gemini_api_key")
    stores = sem_meta.get("stores", [])
    for store_name in list(stores):
        if store_name in data.get("file_stores", {}):
            fs = data["file_stores"][store_name].get("file_search_store_name")
            try:
                if sem_key and genai is not None:
                    client = init_gemini_client(sem_key)
                    try:
                        client.file_search_stores.delete(
                            name=fs,
                            config={"force": True}
                        )
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                del data["file_stores"][store_name]
            except Exception:
                pass
    folder = UPLOAD_ROOT / department / sem
    if folder.exists():
        try:
            shutil.rmtree(folder)
        except Exception:
            pass
    del data["departments"][department]["sems"][sem]
    save_data(data)
    return {"success": True, "deleted_sem": sem}


@app.post("/admin/departments/{department}/sems/{sem}/stores/create")
def create_store_in_sem(
    department: str,
    sem: str,
    store_name: str = Form(...)
):
    data = load_data()

    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")
    if sem not in data["departments"][department].get("sems", {}):
        raise HTTPException(status_code=404, detail="Sem not found")

    sem_meta = data["departments"][department]["sems"][sem]
    sem_key = sem_meta.get("gemini_api_key")

    try:
        client = init_gemini_client(sem_key)
    except Exception as e:
        return JSONResponse(
            {"error": f"Invalid Gemini API key for sem: {e}"},
            status_code=400
        )

    try:
        fs_store = client.file_search_stores.create(
            config={"display_name": store_name}
        )
        fs_store_name = getattr(fs_store, "name", None) or fs_store
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to create File Search store on Gemini: {e}"},
            status_code=500
        )

    file_stores = data.setdefault("file_stores", {})
    if store_name in file_stores:
        return JSONResponse(
            {"error": "A local store with this name already exists"},
            status_code=400
        )

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    file_entry = {
        "store_name": store_name,
        "file_search_store_name": fs_store_name,
        "created_at": created_at,
        "files": [],
        "total_size_bytes": 0
    }
    file_stores[store_name] = file_entry
    sem_meta.setdefault("stores", []).append(store_name)
    save_data(data)
    return {
        "success": True,
        "store_name": store_name,
        "file_search_store_resource": fs_store_name,
        "created_at": created_at
    }


@app.get("/departments/{department}/sems/{sem}/stores")
def list_stores_in_sem(department: str, sem: str):
    data = load_data()
    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")
    if sem not in data["departments"][department].get("sems", {}):
        raise HTTPException(status_code=404, detail="Sem not found")
    stores = data["departments"][department]["sems"][sem].get("stores", [])
    file_stores = data.get("file_stores", {})
    result = [file_stores.get(s) for s in stores if s in file_stores]
    return {"success": True, "stores": result}


@app.delete("/departments/{department}/sems/{sem}/stores/{store_name}")
def delete_store_scoped(department: str, sem: str, store_name: str):
    data = load_data()
    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")
    if sem not in data["departments"][department].get("sems", {}):
        raise HTTPException(status_code=404, detail="Sem not found")
    if store_name not in data["departments"][department]["sems"][sem].get("stores", []):
        raise HTTPException(status_code=404, detail="Store not found in the given sem")
    if store_name not in data.get("file_stores", {}):
        raise HTTPException(status_code=404, detail="Store metadata missing")

    meta = data["file_stores"][store_name]
    fs_store = meta.get("file_search_store_name")
    sem_key = data["departments"][department]["sems"][sem].get("gemini_api_key")

    try:
        if sem_key and genai is not None:
            client = init_gemini_client(sem_key)
            client.file_search_stores.delete(name=fs_store, config={"force": True})
    except Exception:
        pass

    removed_size = meta.get("total_size_bytes", 0)
    sem_meta = data["departments"][department]["sems"][sem]
    sem_meta["total_size_bytes"] = max(
        0, sem_meta.get("total_size_bytes", 0) - removed_size
    )

    folder = UPLOAD_ROOT / department / sem / store_name
    if folder.exists():
        try:
            shutil.rmtree(folder)
        except Exception:
            pass

    try:
        sem_meta["stores"].remove(store_name)
    except Exception:
        pass
    del data["file_stores"][store_name]
    save_data(data)
    return {
        "success": True,
        "deleted_store": store_name,
        "removed_size_bytes": removed_size
    }


# ---------------- Upload files with OCR → TXT and index into Gemini ----------------

@app.post("/departments/{department}/sems/{sem}/stores/{store_name}/upload")
async def upload_files_scoped(
    department: str,
    sem: str,
    store_name: str,
    limit: Optional[bool] = Form(True),
    file_type: str = Form(...),             # pdf / image / docx
    files: List[UploadFile] = File(...)
):
    """
    Upload files into a store, but:
    - Convert each original file (PDF / image / DOCX) into plain TEXT (.txt)
    - Upload ONLY the .txt file to Gemini File Search with mime_type = text/plain
    - Delete local temp files after upload
    - Store only metadata (no persistent content) for student privacy
    """
    data = load_data()

    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")
    if sem not in data["departments"][department].get("sems", {}):
        raise HTTPException(status_code=404, detail="Sem not found")
    if store_name not in data["departments"][department]["sems"][sem].get("stores", []):
        raise HTTPException(status_code=404, detail="Store not found in the given sem")
    if store_name not in data.get("file_stores", {}):
        raise HTTPException(status_code=404, detail="Store metadata missing")

    sem_key = data["departments"][department]["sems"][sem].get("gemini_api_key")
    try:
        client = init_gemini_client(sem_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    store_meta = data["file_stores"][store_name]
    fs_store_name = store_meta.get("file_search_store_name")
    if not fs_store_name:
        raise HTTPException(status_code=500, detail="File Search store mapping missing")

    temp_folder = UPLOAD_ROOT / department / sem / store_name
    temp_folder.mkdir(parents=True, exist_ok=True)

    results = []
    ftype = (file_type or "").lower().strip()

    for upload in files:
        original_filename = upload.filename or "file"
        base_clean = clean_filename(original_filename)
        base_name_no_ext = (
            base_clean.rsplit(".", 1)[0] if "." in base_clean else base_clean
        )
        txt_filename = base_name_no_ext + ".txt"
        temp_path = temp_folder / txt_filename

        # ---- ensure pointer at start, then READ ORIGINAL BYTES ONCE ----
        try:
            await upload.seek(0)
            original_bytes = await upload.read()
            original_size = len(original_bytes)

            # check limit using original file size
            if limit and original_size > MAX_FILE_BYTES:
                results.append({
                    "filename": original_filename,
                    "uploaded": False,
                    "indexed": False,
                    "reason": f"File exceeds limit of {MAX_FILE_BYTES} bytes."
                })
                continue

            # -----------------------------
            # Convert original file → TEXT based on file_type
            # -----------------------------
            if ftype == "pdf":
                text_content = await extract_text_from_pdf(original_bytes)
            elif ftype == "image":
                text_content = await extract_text_from_image(original_bytes)
            elif ftype == "docx":
                text_content = await extract_text_from_docx(original_bytes)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file_type. Expected one of: pdf, image, docx"
                )

            # Save TEXT as .txt file
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(text_content or "")
                size = temp_path.stat().st_size
            except Exception as e:
                results.append({
                    "filename": original_filename,
                    "uploaded": False,
                    "indexed": False,
                    "reason": f"Failed to save local TXT file: {e}"
                })
                continue

        except HTTPException:
            # re-raise HTTPException directly
            raise
        except Exception as e:
            results.append({
                "filename": original_filename,
                "uploaded": False,
                "indexed": False,
                "reason": f"Failed to process file: {e}"
            })
            continue

        # ---------------------------
        # Upload TXT file to Gemini File Search WITH mime_type = text/plain
        # ---------------------------
        document_resource = None
        document_id = None
        indexed_ok = False
        gemini_error = None

        mime = "text/plain"
        upload_config: Dict[str, Any] = {
            "display_name": txt_filename,
            "mime_type": mime
        }

        try:
            op = client.file_search_stores.upload_to_file_search_store(
                file=str(temp_path),
                file_search_store_name=fs_store_name,
                config=upload_config
            )

            op = wait_for_operation(client, op)

            # Try to extract document resource
            try:
                document_resource = op.response.file_search_document.name
            except Exception:
                document_resource = None

            # Fallback: list documents
            if not document_resource:
                docs = rest_list_documents_for_store(fs_store_name, sem_key)
                for d in docs:
                    if d.get("displayName") == txt_filename:
                        document_resource = d.get("name")
                        break

            if document_resource:
                document_id = document_resource.split("/")[-1]
                indexed_ok = True

        except Exception as e:
            gemini_error = str(e)

        # Delete temp TXT file after upload
        try:
            os.remove(temp_path)
        except Exception:
            pass

        # Save metadata only (no file content)
        entry = {
            "display_name": txt_filename,
            "original_filename": original_filename,
            "size_bytes": original_size,  # track original size for usage
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gemini_indexed": indexed_ok,
            "document_resource": document_resource,
            "document_id": document_id,
            "gemini_error": gemini_error,
            "file_type": ftype
        }

        store_meta.setdefault("files", []).append(entry)
        store_meta["total_size_bytes"] += original_size
        data["departments"][department]["sems"][sem]["total_size_bytes"] += original_size
        save_data(data)

        results.append({
            "filename": original_filename,
            "txt_filename": txt_filename,
            "uploaded": True,
            "indexed": indexed_ok,
            "document_resource": document_resource,
            "document_id": document_id,
            "gemini_error": gemini_error,
            "original_size_bytes": original_size
        })

    store_total_bytes = store_meta.get("total_size_bytes", 0)
    sem_total_bytes = data["departments"][department]["sems"][sem].get(
        "total_size_bytes", 0
    )

    return {
        "success": True,
        "results": results,
        "store_total_bytes": store_total_bytes,
        "sem_total_bytes": sem_total_bytes
    }


@app.get("/departments/{department}/sems/{sem}/stores/{store_name}/files")
def list_files_in_store(department: str, sem: str, store_name: str):
    data = load_data()
    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")
    if sem not in data["departments"][department].get("sems", {}):
        raise HTTPException(status_code=404, detail="Sem not found")
    if store_name not in data["departments"][department]["sems"][sem].get("stores", []):
        raise HTTPException(status_code=404, detail="Store not found in the sem")
    if store_name not in data.get("file_stores", {}):
        raise HTTPException(status_code=404, detail="Store metadata missing")
    return {
        "success": True,
        "files": data["file_stores"][store_name].get("files", [])
    }


@app.delete("/departments/{department}/sems/{sem}/stores/{store_name}/documents/{document_id}")
def delete_document_scoped(
    department: str,
    sem: str,
    store_name: str,
    document_id: str
):
    data = load_data()
    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")
    if sem not in data["departments"][department].get("sems", {}):
        raise HTTPException(status_code=404, detail="Sem not found")
    if store_name not in data["departments"][department]["sems"][sem].get("stores", []):
        raise HTTPException(status_code=404, detail="Store not found in the given sem")
    if store_name not in data.get("file_stores", {}):
        raise HTTPException(status_code=404, detail="Store metadata missing")

    meta = data["file_stores"][store_name]
    fs_store = meta.get("file_search_store_name")
    sem_key = data["departments"][department]["sems"][sem].get("gemini_api_key")
    doc_resource = f"{fs_store}/documents/{document_id}"
    url = f"{GEMINI_REST_BASE}/{doc_resource}"
    params = {"force": "true", "key": sem_key}
    try:
        resp = requests.delete(url, params=params, timeout=15)
    except Exception as e:
        return JSONResponse(
            {"success": False, "error": f"REST request failed: {e}"},
            status_code=500
        )
    if resp.status_code not in (200, 204):
        return JSONResponse(
            {"success": False, "error": f"Gemini REST DELETE failed: {resp.text}"},
            status_code=resp.status_code
        )

    removed_size = 0
    new_files = []
    for f in meta.get("files", []):
        if f.get("document_id") == document_id:
            removed_size += f.get("size_bytes", 0)
            continue
        new_files.append(f)
    meta["files"] = new_files
    meta["total_size_bytes"] = max(
        0, meta.get("total_size_bytes", 0) - removed_size
    )
    sem_meta = data["departments"][department]["sems"][sem]
    sem_meta["total_size_bytes"] = max(
        0, sem_meta.get("total_size_bytes", 0) - removed_size
    )
    data["file_stores"][store_name] = meta
    save_data(data)
    return {
        "success": True,
        "deleted_document_id": document_id,
        "removed_size_bytes": removed_size
    }


@app.get("/departments/{department}/sems/{sem}/usage")
def sem_usage(department: str, sem: str):
    data = load_data()
    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")
    if sem not in data["departments"][department].get("sems", {}):
        raise HTTPException(status_code=404, detail="Sem not found")
    sem_meta = data["departments"][department]["sems"][sem]
    stores = sem_meta.get("stores", [])
    file_stores = data.get("file_stores", {})
    store_sizes = {}
    for s in stores:
        if s in file_stores:
            store_sizes[s] = f"{round(file_stores[s].get('total_size_bytes', 0) / (1024 * 1024), 2)} MB"
    sem_total = f"{round(sem_meta.get('total_size_bytes', 0) / (1024 * 1024), 2)} MB"
    return {"success": True, "store_sizes": store_sizes, "sem_total": sem_total}


@app.get("/admin/departments/{department}/usage")
def department_usage(department: str):
    data = load_data()

    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")
    sems = data["departments"][department].get("sems", {})
    sems_sizes = {}
    total = 0
    for sem_name, sem_meta in sems.items():
        b = sem_meta.get("total_size_bytes", 0)
        sems_sizes[sem_name] = f"{round(b / (1024 * 1024), 2)} MB"
        total += b
    return {
        "success": True,
        "department_total": f"{round(total / (1024 * 1024), 2)} MB",
        "sems": sems_sizes
    }


@app.get("/health")
def health_check():
    return {"success": True, "status": "ok"}


# ---------------- Startup quick writable-check ----------------
try:
    test_path = UPLOAD_ROOT / ".write_test"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("ok")
    test_path.unlink()
except Exception as e:
    logger.error(
        "Upload root not writable or cannot create test file: %s (UPLOAD_ROOT=%s)",
        e,
        UPLOAD_ROOT,
    )
# ============================
# smartbvb.py  (PART 3 / 3)
# ============================

# ---------------- Gemini extraction / selection / RAG helpers ----------------

def _extract_system_and_query_sync(client, raw_text: str) -> str:
    system_prompt = (
        "You are a parser. Extract ONLY two things from the student's message as valid JSON:\n"
        "1) system_prompt: any instructions that tell the assistant HOW to behave "
        "(tone/style/format/constraints).\n"
        "2) user_query: the actual academic question to be answered.\n"
        'Return EXACT JSON: {"system_prompt": "...", "user_query": "..."}.\n'
    )
    model = "gemini-2.5-flash"
    response = client.models.generate_content(
        model=model,
        contents=raw_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0
        )
    )
    txt = getattr(response, "text", None)
    if not txt and getattr(response, "candidates", None):
        c = response.candidates[0]
        txt = getattr(c, "text", None) or getattr(c, "content", None)
    return txt or ""


def _parse_json_loose(raw: str) -> Dict:
    raw = (raw or "").strip()
    if not raw:
        return {"system_prompt": "", "user_query": ""}
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                return {"system_prompt": "", "user_query": raw}
        return {"system_prompt": "", "user_query": raw}


def _call_gemini_store_selector_sync(client, stores: List[str], question: str) -> str:
    """
    Ask Gemini to pick which stores (subjects/chapters) are relevant for this student's question.
    It can also split the question per store, and mark any parts that cannot be answered.
    """
    model = "gemini-2.5-flash"
    system_prompt = f"""
You are an academic router that reads a student's question and decides which study-material
stores are relevant.

Each store typically corresponds to a subject, semester, or carefully grouped notes.

Available stores: {stores}

The student question is: "{question}"

You must:

1. Identify which stores are relevant based on:
   - chapter names (e.g., "Chapter 1", "Unit 2", "Module 3")
   - subject names (e.g., "OS", "DBMS", "CN")
   - hints like "according to class notes", "as per PPT", etc.

2. If the question is broad (e.g., “Explain entire Unit 1” or “give all review question solutions for Chapter 1”),
   select every store that might contain that chapter/unit.

3. For each selected store, create a focused question that includes:
   - the chapter/unit/module name (if mentioned)
   - the student's intent (explain, summarize, list definitions, write short notes, solutions, etc.)

Return VALID JSON in this EXACT structure:

{{
  "stores": ["store_name_1", "store_name_2"],
  "split_questions": {{
      "store_name_1": "focused question for this store",
      "store_name_2": "another focused question"
  }},
  "unanswered": [
      {{
        "text": "any remaining part of the query that no store can handle",
        "reason": "why it cannot be answered from given stores"
      }}
  ]
}}

If none of the stores match, return "stores": [] and in "unanswered" put the original question
with a helpful reason.
"""
    response = client.models.generate_content(
        model=model,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0
        )
    )
    txt = getattr(response, "text", None)
    if not txt and getattr(response, "candidates", None):
        c = response.candidates[0]
        txt = getattr(c, "text", None) or getattr(c, "content", None)
    return txt or ""


async def _call_gemini_for_store_selection(
    stores: List[str],
    question: str,
    sem_key: str
) -> Dict:
    if not stores:
        return {"stores": [], "split_questions": {}, "unanswered": []}
    if genai is None:
        return {"stores": stores, "split_questions": {}, "unanswered": []}
    loop = asyncio.get_running_loop()
    try:
        client = await loop.run_in_executor(None, init_gemini_client, sem_key)
        raw = await loop.run_in_executor(
            None,
            _call_gemini_store_selector_sync,
            client,
            stores,
            question
        )
        if not raw:
            return {"stores": stores, "split_questions": {}, "unanswered": []}
        parsed = _parse_json_loose(raw)
        return {
            "stores": parsed.get("stores", []) or [],
            "split_questions": parsed.get("split_questions", {}) or {},
            "unanswered": parsed.get("unanswered", []) or []
        }
    except Exception:
        return {"stores": stores, "split_questions": {}, "unanswered": []}


def data_store_name_for(local_store_name: str) -> str:
    d = load_data()
    return d.get("file_stores", {}).get(local_store_name, {}).get(
        "file_search_store_name", ""
    )


async def _call_rag_for_store(
    sem_key: str,
    store: str,
    question: str,
    system_prompt: Optional[str] = None
) -> Dict:
    """
    Core RAG call for one store.

    For university students:
    - Use File Search documents as *grounding*.
    - BUT Gemini is allowed to extend with its own reasoning to explain concepts,
      summarize chapters, and generate solutions to review questions.
    """
    if genai is None:
        return {"error": True, "detail": "Gemini SDK missing"}
    loop = asyncio.get_running_loop()

    def _sync_call():
        client = init_gemini_client(sem_key)

        # ---- NEW ACADEMIC DEFAULT PROMPT ----
        cfg_prompt = system_prompt or (
            "You are an academic assistant helping university students with notes, PPTs, "
            "lesson plans, and review questions.\n"
            "Use the retrieved File Search documents as your *main reference context*, "
            "but you ARE allowed to extend with your own reasoning and knowledge.\n"
            "\n"
            "Rules:\n"
            "1. If the student asks for 'solutions', 'answers', or 'explain/review questions',\n"
            "   you MUST generate clear, step-by-step solutions for each question found in the context.\n"
            "2. If a chapter/unit/module is mentioned (e.g., 'Chapter 1', 'Unit 2'),\n"
            "   focus on that part of the syllabus and collect all relevant information.\n"
            "3. Combine content from multiple retrieved chunks logically.\n"
            "4. Use simple, student-friendly language and well-structured formatting\n"
            "   (headings, bullet points, numbered steps) whenever it helps understanding.\n"
            "5. If some specific detail is missing from the notes, still try to answer using your\n"
            "   own knowledge, but clearly mention that the detail is not explicitly present in the material.\n"
            "6. NEVER mention File Search, embeddings, or any internal implementation details.\n"
        )

        file_search_tool = types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[data_store_name_for(store)]
            )
        )
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=cfg_prompt,
                    tools=[file_search_tool],
                    temperature=0.0
                )
            )
            txt = getattr(response, "text", None)
            if not txt and getattr(response, "candidates", None):
                c = response.candidates[0]
                txt = getattr(c, "text", None) or getattr(c, "content", None)
            grounding = None
            if hasattr(response, "candidates") and len(response.candidates) > 0:
                grounding = getattr(response.candidates[0], "grounding_metadata", None)
            return {
                "store": store,
                "question": question,
                "answer": txt or "",
                "grounding": grounding
            }
        except Exception as e:
            return {
                "store": store,
                "question": question,
                "answer": "",
                "error": str(e)
            }

    return await loop.run_in_executor(None, _sync_call)


def _merge_answers_apply_system(
    sem_key: str,
    system_prompt: str,
    final_answers: List[Dict]
) -> str:
    """
    Merge per-store answers into one final student-friendly answer.

    We keep the same 'no Q/A storage' guarantee (merge happens in-memory only).
    """
    if not final_answers:
        return ""
    combined = []
    for fa in final_answers:
        s = fa.get("store")
        q = fa.get("question")
        a = fa.get("answer") or ""
        combined.append(f"STORE: {s}\nQUESTION: {q}\nANSWER: {a}\n---")
    combined_text = "\n".join(combined)

    merge_instr = (
        "You are an academic assistant combining answers from multiple study-material stores "
        "into ONE final response for a student.\n"
        "\n"
        "System instructions for the final answer (from the user or application):\n"
        f"{system_prompt}\n"
        "\n"
        "You are given answers from multiple stores below. Your job:\n"
        "1. Merge them into a single coherent explanation.\n"
        "2. Remove duplicates and contradictions.\n"
        "3. Make the final answer well-structured and exam-friendly.\n"
        "4. If the question involved 'review questions' or 'chapter solutions',\n"
        "   ensure that all solutions are presented in an organized way (Q1, Q2, etc.).\n"
        "5. If some information is clearly missing, briefly mention that.\n"
        "6. Do NOT mention stores, File Search, or internal implementation details.\n"
        "\n"
        "Return ONLY the final merged answer text, nothing else.\n"
    )

    try:
        client = init_gemini_client(sem_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=combined_text,
            config=types.GenerateContentConfig(
                system_instruction=merge_instr,
                temperature=0.0
            )
        )
        txt = getattr(response, "text", None)
        if not txt and getattr(response, "candidates", None):
            c = response.candidates[0]
            txt = getattr(c, "text", None) or getattr(c, "content", None)
        return txt or ""
    except Exception:
        parts = []
        for fa in final_answers:
            parts.append(f"{fa.get('store')}: {fa.get('answer')}")
        return "\n\n".join(parts)


# =====================================================
# ASK endpoint implementation (no session, no storing Q/A)
# =====================================================

@app.post("/departments/{department}/sems/{sem}/ask")
async def ask_department_sem(department: str, sem: str, payload: Dict[str, Any]):
    """
    Main endpoint students will call.

    Body: { "question": "..." }

    Flow:
    1. Extract (system_prompt, user_query) from the student's message.
    2. Use Gemini to select which stores (subjects/chapters) are relevant.
    3. Run RAG for each selected store.
    4. If only one store and system_prompt exists, use that single RAG answer directly.
    5. If multiple stores, merge answers and apply the system_prompt at final merge.
    6. Return only final answer + minimal source texts (no Q/A stored).
    """
    question = payload.get("question") if isinstance(payload, dict) else None
    if not question:
        raise HTTPException(status_code=400, detail="question required in request body")

    data = load_data()
    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")
    if sem not in data["departments"][department].get("sems", {}):
        raise HTTPException(status_code=404, detail="Sem not found")

    sem_meta = data["departments"][department]["sems"][sem]
    stores = sem_meta.get("stores", [])
    sem_key = sem_meta.get("gemini_api_key")
    if not sem_key:
        raise HTTPException(status_code=500, detail="Sem Gemini API key missing")

    # Step 1: extract system_prompt + user_query
    system_prompt = ""
    user_query = question
    if genai is not None:
        try:
            loop = asyncio.get_running_loop()
            client = await loop.run_in_executor(None, init_gemini_client, sem_key)
            raw = await loop.run_in_executor(
                None, _extract_system_and_query_sync, client, question
            )
            parsed = _parse_json_loose(raw)
            system_prompt = parsed.get("system_prompt", "") or ""
            user_query = parsed.get("user_query", "") or question
        except Exception:
            system_prompt = ""
            user_query = question

    # Step 2: store selection
    selector = await _call_gemini_for_store_selection(stores, user_query, sem_key)
    selected_stores = selector.get("stores", []) or []
    split_q = selector.get("split_questions", {}) or {}
    unanswered_parts = selector.get("unanswered", []) or []

    if not selected_stores:
        resp_text = "Sorry, your question does not match any available study-material store."
        if unanswered_parts:
            extra = []
            for part in unanswered_parts:
                txt = part.get("text") or ""
                reason = part.get("reason") or ""
                if txt:
                    extra.append(f'- "{txt}" ({reason})' if reason else f'- "{txt}"')
            if extra:
                resp_text += "\n\nDetails:\n" + "\n".join(extra)
        return {
            "success": True,
            "response": resp_text,
            "stores_used": [],
            "sources": [],
            "unanswered_parts": unanswered_parts
        }

    # Step 3: per-store RAG calls
    send_system_to_rag = len(selected_stores) == 1 and bool(system_prompt.strip())
    tasks = []
    for store in selected_stores:
        q = split_q.get(store, user_query)
        tasks.append(
            asyncio.create_task(
                _call_rag_for_store(
                    sem_key,
                    store,
                    q,
                    system_prompt if send_system_to_rag else None
                )
            )
        )

    results = await asyncio.gather(*tasks)

    final_answers = []
    all_sources = []
    for r in results:
        if r.get("error"):
            answer_text = "Sorry, I could not retrieve information from this store."
            sources = []
        else:
            answer_text = r.get("answer", "")
            grounding = r.get("grounding") or {}
            chunks = grounding.get("groundingChunks", []) if isinstance(grounding, dict) else []
            sources = []
            for c in chunks:
                ctx = c.get("retrievedContext", {})
                if ctx.get("text"):
                    # We only keep minimal text as 'source' for transparency;
                    # still we do NOT store question/answer history anywhere.
                    sources.append(ctx["text"])
        final_answers.append({
            "store": r.get("store"),
            "question": r.get("question"),
            "answer": answer_text,
            "sources": sources
        })
        all_sources.extend(sources)

    # Step 4: merge
    if len(final_answers) == 1 and send_system_to_rag:
        merged_text = final_answers[0]["answer"]
    else:
        if system_prompt.strip():
            merged_text = _merge_answers_apply_system(sem_key, system_prompt, final_answers)
        else:
            # No explicit system prompt: just combine store answers plainly but still student-friendly.
            parts = []
            for fa in final_answers:
                parts.append(f"**{fa['store']}**:\n{fa['answer']}")
            merged_text = "\n\n".join(parts)

    # If there were parts that no store could answer, append that info.
    if unanswered_parts:
        extra_lines = []
        for part in unanswered_parts:
            txt = part.get("text") or ""
            reason = part.get("reason") or ""
            if not txt:
                continue
            if reason:
                extra_lines.append(f'- "{txt}" ({reason})')
            else:
                extra_lines.append(f'- "{txt}"')
        if extra_lines:
            merged_text += (
                "\n\nThe following parts of your question could not be fully answered "
                "from the available notes/stores:\n"
                + "\n".join(extra_lines)
            )

    return {
        "success": True,
        "response": merged_text,
        "stores_used": selected_stores,
        "sources": all_sources[:1],  # keep just 1 minimal source snippet
        "unanswered_parts": unanswered_parts
    }


# ---------------- Run guidance ----------------

if __name__ == "__main__":
    print(
        "This module is intended to be run with Uvicorn, e.g.:\n"
        "    uvicorn smartbvb:app --host 0.0.0.0 --port 8000"
    )
