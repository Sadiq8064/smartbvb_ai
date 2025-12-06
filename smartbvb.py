# ============================
# smartbvb.py  (PART 1 / 2)
# ============================
"""
FastAPI backend (smartbvb) — clean production/testing-ready single-file API.

University use-case:

- Admin / Departments / Semesters / Stores
- Each Sem stores a Gemini API key (used for all operations under that sem)
- Upload / delete files (PDF / images / DOCX) → OCR → text → index into Gemini File Search
- ASK endpoint:
    - Student sends: question + list of store names + optional system_prompt
    - We call Gemini File Search directly on those stores (NO prediction / routing)

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
# smartbvb.py  (PART 2 / 2)
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

# =====================================================
# SIMPLE ASK endpoint implementation (NO prediction)
# =====================================================

@app.post("/departments/{department}/sems/{sem}/ask")
def ask_department_sem(department: str, sem: str, payload: Dict[str, Any]):
    """
    Main endpoint students will call.

    Body:
    {
      "question": "your question",
      "stores": ["store1", "store2"],      # local store names under this sem
      "system_prompt": "optional extra instructions"
    }

    Behaviour:
    - Uses the sem's stored Gemini API key.
    - Converts provided local store names into Gemini File Search store names.
    - Calls Gemini ONCE with File Search tool across those stores.
    - Returns final answer text + optional grounding metadata.
    - NO system_prompt extraction. NO store prediction. NO per-store merging.
    """
    if types is None:
        raise HTTPException(
            status_code=500,
            detail="google-genai types not available on server."
        )

    question = payload.get("question") if isinstance(payload, dict) else None
    stores_from_body = payload.get("stores") if isinstance(payload, dict) else None
    system_prompt = payload.get("system_prompt") if isinstance(payload, dict) else None

    if not question:
        raise HTTPException(status_code=400, detail="question required in request body")

    if not stores_from_body or not isinstance(stores_from_body, list):
        raise HTTPException(
            status_code=400,
            detail="stores (list of store names) is required in request body"
        )

    data = load_data()
    if department not in data.get("departments", {}):
        raise HTTPException(status_code=404, detail="Department not found")
    if sem not in data["departments"][department].get("sems", {}):
        raise HTTPException(status_code=404, detail="Sem not found")

    sem_meta = data["departments"][department]["sems"][sem]
    sem_key = sem_meta.get("gemini_api_key")
    if not sem_key:
        raise HTTPException(status_code=500, detail="Sem Gemini API key missing")

    # Convert local store names → Gemini File Search store resource names,
    # but only if they belong to this sem.
    valid_sem_stores = set(sem_meta.get("stores", []))
    file_stores_meta = data.get("file_stores", {})
    fs_store_names: List[str] = []

    for s in stores_from_body:
        if s in valid_sem_stores and s in file_stores_meta:
            fs_name = file_stores_meta[s].get("file_search_store_name")
            if fs_name:
                fs_store_names.append(fs_name)

    if not fs_store_names:
        raise HTTPException(
            status_code=400,
            detail="No valid File Search stores found for the provided store names under this sem."
        )

    # Default system prompt (if none provided)
    system_instruction = system_prompt or (
        "You are an academic assistant for university students. "
        "Use ONLY the provided File Search documents to answer clearly, "
        "with step-by-step, exam-focused explanations where needed."
    )

    try:
        client = init_gemini_client(sem_key)

        file_search_tool = types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=fs_store_names
            )
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[file_search_tool],
                temperature=0.2
            )
        )

        grounding = None
        if hasattr(response, "candidates") and response.candidates:
            grounding = getattr(response.candidates[0], "grounding_metadata", None)

        answer_text = getattr(response, "text", "") or ""

        return {
            "success": True,
            "response": answer_text,
            "stores_used": stores_from_body,
            "grounding_metadata": grounding
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini File Search RAG call failed: {e}"
        )


# ---------------- Run guidance ----------------

if __name__ == "__main__":
    print(
        "This module is intended to be run with Uvicorn, e.g.:\n"
        "    uvicorn smartbvb:app --host 0.0.0.0 --port 8000"
    )
