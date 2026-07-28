"""
Manga Translator
-----------------
Give it an image URL (a manga page), and it will:
  1. Download the image
  2. Detect text regions on the page (EasyOCR's CRAFT detector)
  3. Recognize the Japanese text in each region using manga-ocr, a model
     specifically trained on manga fonts/vertical text (much more accurate
     than generic OCR on manga pages)
  4. Translate each line into English (or any target language)
  5. Paint over the original text and draw the translated text back into
     the same spot on the page
  6. Save the result as a new image, plus print a text-only report

Usage:
    python manga_translator.py <image_url> [--target en] [--out result.png]
    python manga_translator.py <image_url> --engine easyocr   (fallback, no manga-ocr)

First run downloads models (EasyOCR detector + manga-ocr, ~500MB total).
This needs a normal internet connection with access to github.com and
huggingface.co - it will NOT work in a locked-down sandbox.
"""

import argparse
import io
import sys
import textwrap
import time

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
import easyocr
from deep_translator import GoogleTranslator

# manga_ocr is optional - the script still works with plain easyocr if it's
# not installed, just with lower recognition quality on manga-style text.
try:
    from manga_ocr import MangaOcr
    _HAS_MANGA_OCR = True
except ImportError:
    _HAS_MANGA_OCR = False

_mocr_singleton = None


def get_manga_ocr():
    global _mocr_singleton
    if _mocr_singleton is None:
        print("      loading manga-ocr model (first call only)...")
        _mocr_singleton = MangaOcr()
    return _mocr_singleton


# ----------------------------------------------------------------------
# Step 1: fetch the image
# ----------------------------------------------------------------------
def load_image_from_url(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0 (manga-translator-script)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type:
        raise ValueError(
            f"The URL did not return an image (got Content-Type: '{content_type}').\n"
            f"This usually means the link points to a WEBPAGE that *contains* an image, "
            f"not the image file itself.\n"
            f"Fix: open the page, right-click the actual image, choose "
            f"'Copy image address', and use that link instead (it should end in "
            f".jpg/.png/.webp and open directly as a picture when pasted in a browser)."
        )

    try:
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise ValueError(
            f"Downloaded the URL but couldn't read it as an image ({e}). "
            f"Double check it's a direct image link."
        ) from e
    return img


# ----------------------------------------------------------------------
# Step 2: detect text regions (boxes only - no recognition yet)
# ----------------------------------------------------------------------
def detect_boxes(image: Image.Image, reader: easyocr.Reader,
                  min_area_frac=0.0002, max_area_frac=0.15):
    """
    Uses EasyOCR purely as a text-region DETECTOR (not a recognizer).
    Returns a list of (x1, y1, x2, y2) rectangles, filtered to drop
    obviously-wrong detections (near-empty slivers, or huge regions that
    are almost certainly the detector getting confused by manga art).
    """
    np_img = np.array(image)
    img_w, img_h = image.size
    img_area = img_w * img_h

    horizontal_list, free_list = reader.detect(np_img)
    boxes = []

    # horizontal_list entries: [x_min, x_max, y_min, y_max]
    for x_min, x_max, y_min, y_max in horizontal_list[0]:
        boxes.append((x_min, y_min, x_max, y_max))

    # free_list entries: quadrilateral point lists -> convert to rect
    for quad in free_list[0]:
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))

    filtered = []
    for (x1, y1, x2, y2) in boxes:
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        area_frac = (w * h) / img_area
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue  # too tiny (noise) or too huge (bad detection)
        filtered.append((int(x1), int(y1), int(x2), int(y2)))

    return filtered


# ----------------------------------------------------------------------
# Step 3: recognize text in each box
# ----------------------------------------------------------------------
def recognize_box(image: Image.Image, box, reader: easyocr.Reader,
                   engine: str, min_conf: float = 0.35):
    x1, y1, x2, y2 = box
    pad = 4
    crop = image.crop((max(0, x1 - pad), max(0, y1 - pad),
                        x2 + pad, y2 + pad))

    if engine == "manga-ocr" and _HAS_MANGA_OCR:
        mocr = get_manga_ocr()
        text = mocr(crop).strip()
        return text  # manga-ocr doesn't return a confidence score

    # fallback: easyocr's own recognizer, with confidence filtering
    np_crop = np.array(crop)
    results = reader.recognize(np_crop)
    if not results:
        return ""
    _, text, conf = results[0]
    if conf < min_conf:
        return ""
    return text.strip()


# ----------------------------------------------------------------------
# Step 4: translate
# ----------------------------------------------------------------------
def translate_text(text: str, target: str = "en", source: str = "ja",
                    max_retries: int = 3) -> str:
    if not text.strip():
        return ""
    for attempt in range(max_retries):
        try:
            result = GoogleTranslator(source=source, target=target).translate(text)
            if result is None:
                return text
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                print(f"  [!] translation failed for '{text}': {e}", file=sys.stderr)
                return text


def translate_all(texts, target: str = "en", source: str = "ja") -> list:
    """
    Translates a whole list of lines in as few network round-trips as
    possible, instead of one request per line. Hitting the free Google
    Translate endpoint once per bubble (a manga page can easily have 15-20)
    tends to trigger silent rate-limiting: some requests come back None,
    others come back truncated/garbled. Batching avoids most of that, with
    a safe per-line fallback if the batch call itself fails.
    """
    if not texts:
        return []
    try:
        results = GoogleTranslator(source=source, target=target).translate_batch(texts)
        # guard against any individual None entries in the batch result
        return [r if r else orig for r, orig in zip(results, texts)]
    except Exception as e:
        print(f"  [!] batch translation failed ({e}), falling back to one-by-one "
              f"(slower, but more resilient)", file=sys.stderr)
        out = []
        for t in texts:
            out.append(translate_text(t, target=target, source=source))
            time.sleep(0.3)  # small delay to avoid re-triggering rate limits
        return out


# ----------------------------------------------------------------------
# Step 5: redraw the page with translated text in place of the original
# ----------------------------------------------------------------------
def pick_font_size(box, text, font_path=None, max_size=40, min_size=10):
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    size = max_size
    while size > min_size:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        wrapped = textwrap.wrap(text, width=max(1, int(box_w / (size * 0.55))))
        line_h = size * 1.15
        total_h = line_h * max(1, len(wrapped))
        if total_h <= box_h + 10:
            return font, wrapped, size
        size -= 2
    font = ImageFont.truetype(font_path, min_size) if font_path else ImageFont.load_default()
    wrapped = textwrap.wrap(text, width=max(1, int(box_w / (min_size * 0.55))))
    return font, wrapped, min_size


def redraw_page(image: Image.Image, boxes, translations, font_path=None):
    out = image.copy()
    draw = ImageDraw.Draw(out)

    for box, translated in zip(boxes, translations):
        if not translated.strip():
            continue
        x1, y1, x2, y2 = box

        draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], fill="white")

        font, wrapped_lines, size = pick_font_size(box, translated, font_path)
        line_h = size * 1.15
        total_h = line_h * len(wrapped_lines)
        start_y = y1 + max(0, ((y2 - y1) - total_h) / 2)

        for i, line in enumerate(wrapped_lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
            text_x = x1 + max(0, ((x2 - x1) - line_w) / 2)
            text_y = start_y + i * line_h
            draw.text((text_x, text_y), line, fill="black", font=font)

    return out


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def translate_manga_page(url: str, target: str = "en", out_path: str = "translated.png",
                          font_path=None, engine: str = "manga-ocr"):
    if engine == "manga-ocr" and not _HAS_MANGA_OCR:
        print("      [!] manga-ocr not installed, falling back to easyocr recognizer.\n"
              "          Install with: pip install manga-ocr", file=sys.stderr)
        engine = "easyocr"

    print(f"[1/6] Downloading image from: {url}")
    image = load_image_from_url(url)

    print("[2/6] Loading EasyOCR detector (first run downloads models)")
    reader = easyocr.Reader(["ja", "en"], gpu=False)

    print("[3/6] Detecting text regions")
    boxes = detect_boxes(image, reader)
    print(f"      -> Found {len(boxes)} candidate region(s) after filtering")

    print(f"[4/6] Recognizing text (engine={engine})")
    kept_boxes = []
    recognized_texts = []
    for box in boxes:
        text = recognize_box(image, box, reader, engine=engine)
        if not text:
            continue
        kept_boxes.append(box)
        recognized_texts.append(text)

    print(f"[5/6] Translating {len(recognized_texts)} line(s) as a single batch "
          f"(more reliable than one request per line)")
    translations = translate_all(recognized_texts, target=target)
    for original, translated in zip(recognized_texts, translations):
        print(f"      JP: {original!r:35s}  ->  {target.upper()}: {translated!r}")

    print(f"[6/6] Rendering translated page -> {out_path}")
    result_img = redraw_page(image, kept_boxes, translations, font_path=font_path)
    result_img.save(out_path)
    print("Done.")
    return kept_boxes, translations, result_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate a manga page from an image URL.")
    parser.add_argument("url", help="URL of the manga page image")
    parser.add_argument("--target", default="en", help="Target language code (default: en)")
    parser.add_argument("--out", default="translated.png", help="Output image path")
    parser.add_argument("--font", default=None, help="Path to a .ttf font file (recommended for non-English targets)")
    parser.add_argument("--engine", default="manga-ocr", choices=["manga-ocr", "easyocr"],
                         help="Recognition engine (manga-ocr is much more accurate on manga text)")
    args = parser.parse_args()

    translate_manga_page(args.url, target=args.target, out_path=args.out,
                          font_path=args.font, engine=args.engine)