import os
import io
import time
import uuid
import gc
import base64

from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2

cv2.setNumThreads(1)

try:
    if os.environ.get("DISABLE_REMBG", "0") == "1":
        raise ImportError("rembg disabled")
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

ALLOWED_EXT   = {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"}
MAX_CONTENT   = 16 * 1024 * 1024
TARGET_SIZE   = 600
MAX_BYTES     = 240 * 1024

# تقليل النسبة إلى 55% لتبدو الصورة طبيعية أكثر
HEAD_TARGET   = 0.55 
HEAD_TOP_POS  = 0.10 

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT
app.json.ensure_ascii = False

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(bgr):
    if bgr is None or bgr.size == 0: return None
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
    if not len(faces): return None
    faces.sort(key=lambda r: r[2] * r[3], reverse=True)
    return int(faces[0][0]), int(faces[0][1]), int(faces[0][2]), int(faces[0][3])

def analyze_background(bgr):
    h, w = bgr.shape[:2]
    patch = max(10, min(h, w) // 10)
    corners = [bgr[0:patch, 0:patch], bgr[0:patch, w - patch:w], bgr[h - patch:h, 0:patch], bgr[h - patch:h, w - patch:w]]
    means = [float(c.reshape(-1, 3).astype(np.float32).mean()) for c in corners]
    stds = [float(c.reshape(-1, 3).astype(np.float32).std()) for c in corners]
    border_std = float(np.concatenate([c.reshape(-1, 3) for c in corners]).astype(np.float32).std())
    return {
        "is_white": (np.mean(means) >= 220 and max(stds) <= 25),
        "has_shadows": (border_std > 20 or max(stds) > 20),
    }

def make_background_white(img_rgb):
    arr = np.array(img_rgb)
    h, w = arr.shape[:2]
    patch = max(10, min(h, w) // 10)
    corners = np.vstack([arr[0:patch, 0:patch].reshape(-1, 3), arr[0:patch, w-patch:w].reshape(-1, 3)])
    bg_color = np.mean(corners, axis=0)
    diff = np.linalg.norm(arr.astype(np.float32) - bg_color, axis=2)
    mask = diff < 45
    mask = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (9, 9), 0)
    alpha = mask.astype(np.float32) / 255.0
    white_bg = np.ones_like(arr, dtype=np.float32) * 255.0
    out = arr.astype(np.float32) * (1 - alpha[..., None]) + white_bg * alpha[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def crop_square_with_padding(img, left, top, size):
    h, w = img.shape[:2]
    left, top, size = int(round(left)), int(round(top)), int(round(size))
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, left + size - w), max(0, top + size - h)
    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        left += pad_l; top += pad_t
    return img[top:top + size, left:left + size]

def enhance_lighting(bgr):
    if bgr is None or bgr.size == 0: return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 45)
    target = max(float(np.percentile(blur, 80)), 1.0)
    gain = np.clip(target / np.maximum(blur, 1.0), 0.8, 1.5)
    gain = 1.0 + (gain - 1.0) * 0.6
    out = np.clip(bgr.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def encode_jpeg_under_limit(pil_img, limit=MAX_BYTES):
    for q in (95, 92, 90, 87, 84, 80, 76, 72, 68, 64, 60, 55, 50, 45, 40, 35, 30, 25, 20):
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        if len(data) <= limit: return data, q
    return data, 20

def draw_guides(pil_img, face_box):
    """رسم الخطوط الإرشادية على الصورة لإثبات التعديل"""
    draw = ImageDraw.Draw(pil_img)
    
    if face_box:
        x, y, w, h = face_box
        head_top = int(y - 0.5 * h)
        head_bottom = int(y + 1.0 * h) # الذقن التقريبي
        eye_level = int(y + 0.5 * h)
        
        # خط أحمر لبداية الرأس
        draw.line([(0, head_top), (TARGET_SIZE, head_top)], fill="red", width=3)
        # خط أحمر لنهاية الذقن
        draw.line([(0, head_bottom), (TARGET_SIZE, head_bottom)], fill="red", width=3)
        # خط أخضر لمستوى العينين
        draw.line([(0, eye_level), (TARGET_SIZE, eye_level)], fill="lime", width=3)
        
        # رسم مستطيل حول الوجه
        draw.rectangle([x, y, x+w, y+h], outline="blue", width=2)

    # رسم إطار الصورة 600x600
    draw.rectangle([0, 0, TARGET_SIZE-1, TARGET_SIZE-1], outline="black", width=4)
    return pil_img

def process_photo(pil_img, original_bytes_len, original_format):
    try:
        issues, actions = [], []
        img = ImageOps.exif_transpose(pil_img).convert("RGB")
        ow, oh = img.size
        
        fmt = (original_format or "").lower().replace("jpg", "jpeg")
        if fmt != "jpeg": issues.append("not_jpeg")
        if original_bytes_len > MAX_BYTES: issues.append("file_too_large")
        if ow != oh: issues.append("not_square")

        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        bg = analyze_background(bgr)
        
        if not bg["is_white"]:
            issues.append("bg_not_white")
            if bg["has_shadows"]: issues.append("bg_shadows")
            if REMBG_AVAILABLE:
                try:
                    img = rembg_remove(img).convert("RGB")
                    white = Image.new("RGB", img.size, (255, 255, 255))
                    white.paste(img, mask=img.split()[-1])
                    img = white
                except:
                    img = make_background_white(img)
            else:
                img = make_background_white(img)
            actions.append("bg_replaced")

        bgr2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h2, w2 = bgr2.shape[:2]
        face = detect_face(bgr2)

        face_out = None
        if face:
            fx, fy, fw, fh = face
            head_top = fy - 0.5 * fh
            head_height = 1.55 * fh 
            crop_size = max(head_height / HEAD_TARGET, 300)
            crop_size = min(crop_size, max(w2, h2) * 1.5) 
            
            center_x = fx + fw / 2.0
            left = center_x - crop_size / 2.0
            top = head_top - HEAD_TOP_POS * crop_size
            
            crop = crop_square_with_padding(bgr2, left, top, crop_size)
            actions.append("head_adjusted")
            
            # حساب مكان الوجه في الصورة الجديدة 600x600
            s = TARGET_SIZE / float(crop_size)
            face_out = (
                int((fx - left) * s), 
                int((fy - top) * s), 
                int(fw * s), 
                int(fh * s)
            )
        else:
            issues.append("face_not_found")
            side = min(h2, w2)
            crop = crop_square_with_padding(bgr2, (w2 - side) / 2.0, (h2 - side) / 2.0, side)

        del bgr, bgr2, img
        gc.collect()

        crop = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
        crop = enhance_lighting(crop)
        
        crop[0:5, :] = 255
        crop[-5:, :] = 255
        crop[:, 0:5] = 255
        crop[:, -5:] = 255

        final_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        
        # إرسال نسختين: واحدة نظيفة وواحدة بالخطوط
        clean_img = final_img.copy()
        guide_img = draw_guides(final_img.copy(), face_out)

        clean_bytes, quality = encode_jpeg_under_limit(clean_img, MAX_BYTES)
        guide_bytes, _ = encode_jpeg_under_limit(guide_img, MAX_BYTES * 2) # الخطوط يمكن أن تكون أوضح

        actions.append("converted_jpg")
        if original_bytes_len > MAX_BYTES: actions.append("compressed")

        return clean_bytes, guide_bytes, {
            "issues": list(dict.fromkeys(issues)), 
            "actions": list(dict.fromkeys(actions)), 
            "compliant": len(issues) == 0
        }
    finally:
        gc.collect()

@app.route("/")
def index():
    return render_template("index.html", rembg=REMBG_AVAILABLE)

@app.route("/healthz")
def healthz(): return "OK", 200

@app.route("/api/process", methods=["POST"])
def api_process():
    if "photo" not in request.files: return jsonify({"ok": False, "error": "no_file"}), 400
    f = request.files["photo"]
    if not f or f.filename == "": return jsonify({"ok": False, "error": "no_file"}), 400

    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_EXT: return jsonify({"ok": False, "error": "bad_format"}), 400

    raw = f.read()
    if not raw: return jsonify({"ok": False, "error": "no_file"}), 400

    try:
        pil_img = Image.open(io.BytesIO(raw))
        pil_img.load()
        pil_img.thumbnail((800, 800), Image.LANCZOS)
    except Exception:
        return jsonify({"ok": False, "error": "bad_format"}), 400

    try:
        clean_bytes, guide_bytes, report = process_photo(pil_img, len(raw), pil_img.format)
    except Exception as e:
        return jsonify({"ok": False, "error": "server", "detail": str(e)}), 500
    finally:
        del pil_img, raw
        gc.collect()

    clean_b64 = base64.b64encode(clean_bytes).decode('utf-8')
    guide_b64 = base64.b64encode(guide_bytes).decode('utf-8')

    return jsonify({
        "ok": True,
        "result_url": f"data:image/jpeg;base64,{guide_b64}",  # الصورة بالخطوط للمعاينة
        "download_url": f"data:image/jpeg;base64,{clean_b64}", # الصورة النظيفة للتحميل
        "report": report,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
