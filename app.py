# app.py — Flask + Roboflow Hosted Inference with A4 report export
# ------------------------------------------------------------
# How to run:
#   pip install flask inference-sdk pillow opencv-python
#   python app.py
# Then open: http://127.0.0.1:5000/

import os
import math
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from PIL import Image, ImageDraw, ImageFont
import cv2

from inference_sdk import InferenceHTTPClient

# ================== CONFIG ==================
# API_URL = "https://serverless.roboflow.com"
# API_KEY = "122aOY67jDoRdfvlcYg6"
# BASE_MODEL_ID = "tb-all-3-lzjdz/2"

API_URL = "https://serverless.roboflow.com"
API_KEY = "122aOY67jDoRdfvlcYg6"
BASE_MODEL_ID = "tuberculosis-detection-xxmxp/1"

# Thresholds (0.0–1.0)
CONF_THRESHOLD = 0.1
OVERLAP_THRESHOLD = 0.1

# A4 canvas (portrait) at 300 DPI
A4_WIDTH_PX = 2480
A4_HEIGHT_PX = 3508
PAGE_MARGIN = 120  # px

# Folders (under /static so files can be served by Flask easily)
STATIC_DIR = Path("static")
UPLOAD_DIR = STATIC_DIR / "uploads"
OUTPUT_DIR = STATIC_DIR / "outputs"
REPORT_DIR = STATIC_DIR / "reports"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

for d in [STATIC_DIR, UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
# ============================================

app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret-for-flash-messages"

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix in ALLOWED_EXTENSIONS

def load_image_pil(path: Path):
    """Load an image as PIL.Image in RGB."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        arr = cv2.imread(str(path))
        if arr is None:
            raise RuntimeError("Unable to load image.")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr)

def build_model_id_with_params(model_id: str, conf: float, overlap: float) -> str:
    sep = "&" if "?" in model_id else "?"
    return f"{model_id}{sep}confidence={conf}&overlap={overlap}"

def draw_rectangles_only(img: Image.Image, preds: dict) -> Image.Image:
    """Draw rectangles for predictions on the image in-place and return it."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    thickness = max(2, math.ceil(min(w, h) * 0.0025))

    for p in preds.get("predictions", []):
        if p.get("confidence", 0.0) < CONF_THRESHOLD:
            continue
        x, y = p.get("x"), p.get("y")
        bw, bh = p.get("width"), p.get("height")
        if None in (x, y, bw, bh):
            continue
        left = max(0, x - bw / 2)
        top = max(0, y - bh / 2)
        right = min(w - 1, x + bw / 2)
        bottom = min(h - 1, y + bh / 2)
        color = (0, 255, 0)
        for t in range(thickness):
            draw.rectangle([left - t, top - t, right + t, bottom + t], outline=color)
    return img

def count_mtb(preds: dict) -> int:
    """Count predictions above confidence threshold."""
    n = 0
    for p in preds.get("predictions", []):
        if p.get("confidence", 0.0) >= CONF_THRESHOLD:
            n += 1
    return n

def try_load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Attempt to load a common TrueType font; fall back to default if unavailable."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: ImageFont.ImageFont, fill=(0, 0, 0), max_width=A4_WIDTH_PX - 2*PAGE_MARGIN, line_spacing=8):
    """Draw multi-line wrapped text within max_width, returns bottom y of the block."""
    if not text:
        return xy[1]
    words = text.split()
    line = ""
    x, y = xy
    for word in words:
        test = f"{line} {word}".strip()
        w, h = draw.textbbox((0, 0), test, font=font)[2:]
        if w <= max_width:
            line = test
        else:
            draw.text((x, y), line, font=font, fill=fill)
            y += h + line_spacing
            line = word
    if line:
        h = draw.textbbox((0, 0), line, font=font)[3]
        draw.text((x, y), line, font=font, fill=fill)
        y += h
    return y

def make_report(annotated_img_path: Path, export_type: str, patient: dict, mtb_count: int, remarks: str) -> Path:
    """Compose a single-page A4 report image and save as PNG or PDF."""
    # Base page
    page = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), "white")
    draw = ImageDraw.Draw(page)

    # Fonts
    title_font = try_load_font(64)
    h2_font = try_load_font(40)
    body_font = try_load_font(34)
    small_font = try_load_font(28)

    # Title and meta
    x = PAGE_MARGIN
    y = PAGE_MARGIN
    title = "PLEMA – Tuberculosis Detection Report"
    draw.text((x, y), title, font=title_font, fill=(0, 0, 0))
    y += draw.textbbox((0, 0), title, font=title_font)[3] + 20

    # Timestamp only (model info removed)
    meta = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    y = draw_wrapped_text(draw, meta, (x, y), font=small_font, fill=(80, 80, 80))
    y += 20

    # Patient block
    draw.text((x, y), "Patient Details", font=h2_font, fill=(0, 0, 0))
    y += draw.textbbox((0, 0), "Patient Details", font=h2_font)[3] + 10

    pd_lines = [
        f"Name: {patient.get('name', '').strip()}",
        f"Patient ID: {patient.get('id', '').strip()}",
        f"Age: {patient.get('age', '').strip()}",
        f"Sex: {patient.get('sex', '').strip()}",
    ]
    for line in pd_lines:
        draw.text((x, y), line, font=body_font, fill=(0, 0, 0))
        y += draw.textbbox((0, 0), line, font=body_font)[3] + 6

    y += 16

    # Findings
    draw.text((x, y), "Findings", font=h2_font, fill=(0, 0, 0))
    y += draw.textbbox((0, 0), "Findings", font=h2_font)[3] + 10
    finding_line = f"Number of MTB Detected: {mtb_count}"
    draw.text((x, y), finding_line, font=body_font, fill=(0, 0, 0))
    y += draw.textbbox((0, 0), finding_line, font=body_font)[3] + 20

    # Remarks
    draw.text((x, y), "Remarks", font=h2_font, fill=(0, 0, 0))
    y += draw.textbbox((0, 0), "Remarks", font=h2_font)[3] + 10
    y = draw_wrapped_text(draw, remarks or "", (x, y), font=body_font, max_width=A4_WIDTH_PX - 2*PAGE_MARGIN)
    y += 20

    # Annotated image placement
    draw.text((x, y), "Annotated Image", font=h2_font, fill=(0, 0, 0))
    y += draw.textbbox((0, 0), "Annotated Image", font=h2_font)[3] + 10

    try:
        annotated = Image.open(annotated_img_path).convert("RGB")
    except Exception:
        annotated = Image.new("RGB", (800, 600), color="lightgray")
        ImageDraw.Draw(annotated).text((20, 20), "Image unavailable", font=body_font, fill=(0, 0, 0))

    # Fit the image into the remaining space
    max_w = A4_WIDTH_PX - 2 * PAGE_MARGIN
    max_h = A4_HEIGHT_PX - y - PAGE_MARGIN
    aw, ah = annotated.size
    scale = min(max_w / aw, max_h / ah) if aw and ah else 1.0
    new_size = (max(1, int(aw * scale)), max(1, int(ah * scale)))
    annotated_resized = annotated.resize(new_size, Image.LANCZOS)
    page.paste(annotated_resized, (x, y))

    # Output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"report_{timestamp}"
    if export_type == "png":
        report_path = REPORT_DIR / f"{base}.png"
        page.save(report_path, "PNG")
    else:
        report_path = REPORT_DIR / f"{base}.pdf"
        page.save(report_path, "PDF", resolution=300.0)

    return report_path

@app.route("/", methods=["GET", "POST"])
def index():
    orig_url = None
    out_url = None
    report_url = None

    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No image selected.")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload a JPG or PNG.")
            return redirect(request.url)

        patient = {
            "name": request.form.get("patient_name", "").strip(),
            "id": request.form.get("patient_id", "").strip(),
            "age": request.form.get("patient_age", "").strip(),
            "sex": request.form.get("patient_sex", "").strip(),
        }
        remarks = request.form.get("remarks", "").strip()
        export_type = request.form.get("export_type", "pdf").lower()
        if export_type not in {"pdf", "png"}:
            export_type = "pdf"

        filename = secure_filename(file.filename)
        upload_path = UPLOAD_DIR / filename
        file.save(upload_path)

        safe_key = API_KEY.strip().replace("+", "%2B")
        try:
            client = InferenceHTTPClient(api_url=API_URL, api_key=safe_key)
            model_id = build_model_id_with_params(BASE_MODEL_ID, CONF_THRESHOLD, OVERLAP_THRESHOLD)
            result = client.infer(str(upload_path), model_id=model_id)
        except Exception as e:
            flash(f"Inference failed: {e}")
            return redirect(request.url)

        try:
            img = load_image_pil(upload_path)
            img_out = draw_rectangles_only(img, result)
            out_filename = f"{Path(filename).stem}_predicted{Path(filename).suffix.lower()}"
            out_path = OUTPUT_DIR / out_filename
            img_out.save(out_path)
        except Exception as e:
            flash(f"Post-processing failed: {e}")
            return redirect(request.url)

        try:
            mtb_count = count_mtb(result)
            report_path = make_report(out_path, export_type, patient, mtb_count, remarks)
        except Exception as e:
            flash(f"Report generation failed: {e}")
            return redirect(request.url)

        orig_url = url_for("static", filename=f"uploads/{filename}")
        out_url = url_for("static", filename=f"outputs/{out_filename}")
        rel_report = report_path.relative_to(STATIC_DIR).as_posix()
        report_url = url_for("static", filename=rel_report)

    return render_template(
        "index.html",
        orig_url=orig_url,
        out_url=out_url,
        report_url=report_url,
    )

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
