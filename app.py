# =====================================================================
# app.py — DV Lottery Photo Checker & Auto-Fixer (Flask backend)
# =====================================================================
import os
import io
import time
import uuid

from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageOps
import numpy as np
import cv2

# تعطيل الخيوط المتعددة في OpenCV لتقليل استهلاك الذاكرة (مهم جداً لـ Render)
cv2.setNumThreads(1)

# تعطيل rembg إذا تم ضبط متغير البيئة
try:
    if os.environ.get("DISABLE_REMBG", "0") == "1":
        raise ImportError("rembg disabled via DISABLE_REMBG env var")
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
RESULT_DIR = os.path.join(STATIC_DIR, "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ALLOWED_EXT   = {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"}
MAX_CONTENT   = 16 * 1024 * 1024        # 16 MB upload limit
TARGET_SIZE   = 600                     # final width = height = 600
MAX_BYTES     = 240 * 1024              # < 240 KB
HEAD_MIN      = 0.50                    # head 50% ..
HEAD_MAX      = 0.69                    # .. 69% of image height
HEAD_TARGET   = 0.60                    # aim safely in the middle
HEAD_TOP_POS  = 0.15                    # head-top position (15% from top)
MAX_AGE_SEC   = 60 * 60                 # auto-delete generated files after 1h

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT
app.json.ensure_ascii = False

# Haar cascades
FACE_CASCADE    = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

# =====================================================================
#  UTILITIES
# =====================================================================
def cleanup_old_files():
    now = time.time()
    for folder in (UPLOAD_DIR, RESULT_DIR):
        try:
            for name in os.listdir(folder):
                path = os.path.join(folder, name)
                if os.path.isfile(path) and now - os.path.getmtime(path) > MAX_AGE_SEC:
                    os.remove(path)
        except Exception:
            pass

def detect_face(bgr):
    if bgr is None or bgr.size == 0:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape[:2]
    min_side = max(40, min(h, w) // 12)

    faces = []
    for sf in (1.05, 1.08, 1.12, 1.18):
        f = FACE_CASCADE.detectMultiScale(gray, scaleFactor=sf, minNeighbors=5, minSize=(min_side, min_side))
        if len(f):
            faces = list(f)
            break
    if not len(faces):
        return None
    faces.sort(key=lambda r: r[2] * r[3], reverse=True)
    x, y, fw, fh = faces[0]
    return int(x), int(y), int(fw), int(fh)

def analyze_background(bgr):
    h, w = bgr.shape[:2]
    patch = max(10, min(h, w) // 10)
    corners = [bgr[0:patch, 0:patch], bgr[0:patch, w - patch:w], bgr[h - patch:h, 0:patch], bgr[h - patch:h, w - patch:w]]
    means, stds = [], []
    for c in corners:
        px = c.reshape(-1, 3).astype(np.float32)
        means.append(float(px.mean()))
        stds.append(float(px.std()))
    border = np.concatenate([
        bgr[0:patch, :].reshape(-1, 3), bgr[h - patch:h, :].reshape(-1, 3),
        bgr[:, 0:patch].reshape(-1, 3), bgr[:, w - patch:w].reshape(-1, 3),
    ]).astype(np.float32)
    avg_corner   = float(np.mean(means))
    worst_std    = float(max(stds))
    border_std   = float(border.std())
    return {
        "is_white":   (avg_corner >= 235 and worst_std <= 20),
        "has_shadows": (border_std > 20 or worst_std > 20),
    }

def grabcut_alpha(img_rgb):
    arr = np.array(img_rgb)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    rect = (int(w * 0.06), int(h * 0.04), int(w * 0.88), int(h * 0.92))
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return np.full((h, w), 255, np.uint8)
    alpha = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    alpha = cv2.medianBlur(alpha, 5)
    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        clean = np.zeros_like(alpha)
        cv2.drawContours(clean, [big], -1, 255, -1)
        alpha = cv2.bitwise_and(alpha, clean)
    return cv2.GaussianBlur(alpha, (5, 5), 0)

def remove_background_alpha(img_rgb):
    if REMBG_AVAILABLE:
        try:
            out = rembg_remove(img_rgb)
            if out.mode != "RGBA": out = out.convert("RGBA")
            return np.array(out.split()[-1])
        except Exception: pass
    return grabcut_alpha(img_rgb)

def crop_square_with_padding(img, left, top, size):
    h, w = img.shape[:2]
    left, top, size = int(round(left)), int(round(top)), int(round(size))
    pad_l = max(0, -left); pad_t = max(0, -top)
    pad_r = max(0, left + size - w); pad_b = max(0, top + size - h)
    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        left += pad_l; top += pad_t
    return img[top:top + size, left:left + size]

def enhance_lighting(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 45)
    target = max(float(np.percentile(blur, 80)), 1.0)
    gain = target / np.maximum(blur, 1.0)
    gain = np.clip(gain, 0.75, 1.60)
    gain = 1.0 + (gain - 1.0) * 0.60
    out = np.clip(bgr.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    mean_l = float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean())
    gamma = 0.90 if mean_l < 110 else (1.10 if mean_l > 195 else 1.0)
    if gamma != 1.0:
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        out = cv2.LUT(out, lut)
    return out

def encode_jpeg_under_limit(pil_img, limit=MAX_BYTES):
    last, last_q = None, 20
    for q in (95, 92, 90, 87, 84, 80, 76, 72, 68, 64, 60, 55, 50, 45, 40, 35, 30, 25, 20):
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        last, last_q = data, q
        if len(data) <= limit: return data, q
    return last, last_q

# =====================================================================
#  MAIN PIPELINE
# =====================================================================
def process_photo(pil_img, original_bytes_len, original_format):
    issues, actions, metrics = [], [], {}
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    ow, oh = img.size
    metrics["original_width"], metrics["original_height"] = ow, oh
    metrics["original_kb"] = round(original_bytes_len / 1024.0, 1)

    fmt = (original_format or "").lower().replace("jpg", "jpeg")
    if fmt != "jpeg": issues.append("not_jpeg")
    if original_bytes_len > MAX_BYTES: issues.append("file_too_large")
    if ow != oh: issues.append("not_square")
    if ow < TARGET_SIZE or oh < TARGET_SIZE: issues.append("low_resolution")

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    bg = analyze_background(bgr)
    need_bg_fix = not bg["is_white"]
    if need_bg_fix:
        issues.append("bg_not_white")
        if bg["has_shadows"]: issues.append("bg_shadows")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean()); contrast = float(gray.std())
    metrics["brightness"] = round(brightness, 1); metrics["contrast"] = round(contrast, 1)
    lighting_issue = False
    if brightness < 95: issues.append("too_dark"); lighting_issue = True
    elif brightness > 218: issues.append("overexposed"); lighting_issue = True
    if contrast < 32: issues.append("low_contrast"); lighting_issue = True

    alpha_small = None
    if need_bg_fix:
        alpha = remove_background_alpha(img)
        alpha_small = alpha
        white = Image.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=Image.fromarray(alpha))
        img = white; actions.append("bg_replaced")

    bgr2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h2, w2 = bgr2.shape[:2]
    face = detect_face(bgr2)
    metrics["face_detected"] = bool(face)

    if face:
        fx, fy, fw, fh = face
        head_top = fy - 0.5 * fh
        head_height = 1.5 * fh
        ratio_orig = head_height / float(h2)
        metrics["head_ratio_original"] = round(ratio_orig * 100, 1)
        if ratio_orig < HEAD_MIN or ratio_orig > HEAD_MAX: issues.append("head_proportion")
        crop_size = max(head_height / HEAD_TARGET, 260)
        center_x = fx + fw / 2.0
        left = center_x - crop_size / 2.0
        top = head_top - HEAD_TOP_POS * crop_size
        crop = crop_square_with_padding(bgr2, left, top, crop_size)
        s = TARGET_SIZE / float(crop_size)
        face_out = ((fx - left) * s, (fy - top) * s, fw * s, fh * s)
        actions.append("head_adjusted")
    else:
        issues.append("face_not_found")
        side = min(h2, w2)
        left = (w2 - side) / 2.0; top = (h2 - side) / 2.0
        crop = crop_square_with_padding(bgr2, left, top, side)
        face_out = None

    crop = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
    actions.append("resized_600")
    enhanced = enhance_lighting(crop)

    mask = np.zeros((TARGET_SIZE, TARGET_SIZE), np.uint8)
    if face_out:
        bx, by, bw, bh = face_out
        x0, y0 = int(max(0, bx - 0.75 * bw)), int(max(0, by - 0.95 * bh))
        x1, y1 = int(min(TARGET_SIZE, bx + bw + 0.75 * bw)), int(min(TARGET_SIZE, by + bh + 0.60 * bh))
    else:
        x0, y0, x1, y1 = 60, 40, TARGET_SIZE - 60, TARGET_SIZE - 60
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 55).astype(np.float32) / 255.0
    crop = np.clip(enhanced.astype(np.float32) * mask[..., None] + crop.astype(np.float32) * (1.0 - mask[..., None]), 0, 255).astype(np.uint8)
    if lighting_issue: actions.append("lighting_fixed")

    if alpha_small is not None:
        a = cv2.resize(alpha_small, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        a = cv2.erode(a, np.ones((3, 3), np.float32), iterations=1)
        a = cv2.GaussianBlur(a, (0, 0), 1.2)
        crop = np.clip(crop.astype(np.float32) * a[..., None] + 255.0 * (1.0 - a[..., None]), 0, 255).astype(np.uint8)

    final_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    jpeg_bytes, quality = encode_jpeg_under_limit(final_img, MAX_BYTES)
    if original_bytes_len > MAX_BYTES or len(jpeg_bytes) < original_bytes_len: actions.append("compressed")
    actions.append("converted_jpg")

    metrics["final_width"], metrics["final_height"] = TARGET_SIZE, TARGET_SIZE
    metrics["final_kb"] = round(len(jpeg_bytes) / 1024.0, 1)
    metrics["jpeg_quality"] = quality
    metrics["head_ratio_final"] = round(HEAD_TARGET * 100, 1)

    def uniq(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen: seen.add(x); out.append(x)
        return out
    return jpeg_bytes, {"issues": uniq(issues), "actions": uniq(actions), "metrics": metrics, "compliant": len(issues) == 0}

# =====================================================================
#  ROUTES
# =====================================================================
@app.route("/")
def index():
    cleanup_old_files()
    return render_template("index.html", rembg=REMBG_AVAILABLE)

@app.route("/healthz")
def healthz():
    return "OK", 200

@app.route("/api/health")
def health():
    return jsonify({"ok": True, "rembg": REMBG_AVAILABLE, "opencv": cv2.__version__})

@app.route("/api/process", methods=["POST"])
def api_process():
    cleanup_old_files()
    if "photo" not in request.files: return jsonify({"ok": False, "error": "no_file"}), 400
    f = request.files["photo"]
    if not f or f.filename == "": return jsonify({"ok": False, "error": "no_file"}), 400

    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_EXT: return jsonify({"ok": False, "error": "bad_format"}), 400

    raw = f.read()
    if not raw: return jsonify({"ok": False, "error": "no_file"}), 400
    if len(raw) > MAX_CONTENT: return jsonify({"ok": False, "error": "too_big"}), 400

    try:
        pil_img = Image.open(io.BytesIO(raw))
        pil_img.load()
        # تصغير الصورة فوراً لتجنب انهيار الذاكرة في Render
        pil_img.thumbnail((1200, 1200), Image.LANCZOS)
    except Exception:
        return jsonify({"ok": False, "error": "bad_format"}), 400

    uid = uuid.uuid4().hex
    try:
        preview = ImageOps.exif_transpose(pil_img).convert("RGB")
        preview.thumbnail((900, 900), Image.LANCZOS)
        orig_name = f"{uid}_orig.jpg"
        preview.save(os.path.join(UPLOAD_DIR, orig_name), format="JPEG", quality=90, optimize=True)
    except Exception:
        orig_name = None

    try:
        result_bytes, report = process_photo(pil_img, len(raw), pil_img.format)
    except Exception as e:
        app.logger.exception("processing failed")
        return jsonify({"ok": False, "error": "server", "detail": str(e)}), 500

    res_name = f"{uid}_dv600.jpg"
    with open(os.path.join(RESULT_DIR, res_name), "wb") as fh:
        fh.write(result_bytes)

    return jsonify({
        "ok": True,
        "original_url": f"/static/uploads/{orig_name}" if orig_name else None,
        "result_url": f"/static/results/{res_name}",
        "download_url": f"/download/{res_name}",
        "report": report,
    })

@app.route("/download/<path:filename>")
def download(filename):
    safe = os.path.basename(filename)
    path = os.path.join(RESULT_DIR, safe)
    if not os.path.isfile(path): return jsonify({"ok": False, "error": "not_found"}), 404
    return send_from_directory(RESULT_DIR, safe, as_attachment=True, download_name="dv-lottery-photo-600x600.jpg")

@app.errorhandler(413)
def too_large(_e):
    return jsonify({"ok": False, "error": "too_big"}), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
