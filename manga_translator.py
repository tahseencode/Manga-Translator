import io
import os
import sys
import textwrap
import time
from urllib.parse import urljoin

# Console output safety net: printing Japanese text to a non-UTF-8 terminal
# (e.g. the cp1252 Windows console) raises UnicodeEncodeError and can crash a
# streaming request. Serverless runtimes are UTF-8 already - this only guards
# against awkward local environments and never changes UTF-8 behaviour.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
del _stream

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from rapidocr import RapidOCR

from bubble_detector import detect_bubbles

# RapidOCR ships its ONNX models inside the wheel, so nothing needs to be
# downloaded at runtime - this keeps Vercel cold starts fast and the
# bundle tiny compared to the torch/easyocr stack it replaces.
_ocr_engine_singleton = None


def get_ocr_engine():
    """Returns a lazily-created RapidOCR engine (det + rec + cls, ONNX-based)."""
    global _ocr_engine_singleton
    if _ocr_engine_singleton is None:
        print("      loading rapidocr models (first call only)...")
        _ocr_engine_singleton = RapidOCR()
    return _ocr_engine_singleton


def initialize_ocr_models():
    get_ocr_engine()


def _iter_ocr_results(result):
    """
    Yields (box, text, score) tuples from a RapidOCR call result.

    rapidocr v3 changed the result container between releases (plain list of
    [box, text, score] in older versions, an object with .boxes/.txts/.scores
    in newer ones), so we normalise both shapes here.
    """
    if result is None:
        return
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        txts = list(result.txts)
        scores = list(result.scores)
        for i, box in enumerate(boxes):
            score = float(scores[i]) if i < len(scores) else 0.0
            yield box, txts[i], score
    elif isinstance(result, (list, tuple)):
        for item in result:
            if len(item) < 3:
                continue
            yield item[0], item[1], float(item[2])


# ----------------------------------------------------------------------
# Step 1: fetch the image
# ----------------------------------------------------------------------
def load_image_from_url(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0 (manga-translator-script)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type:
        raise ValueError(f"URL does not appear to be a direct image link (Content-Type: {content_type})")

    try:
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise ValueError(
            f"Downloaded the URL but couldn't read it as an image ({e}). "
            f"Double check it's a direct image link."
        ) from e
    return img


def find_image_urls_on_page(page_url: str) -> list:
    """
    Scrapes a webpage to find all direct image URLs that look like manga pages.
    """
    print(f"      Scraping {page_url} for image links...")
    headers = {"User-Agent": "Mozilla/5.0 (manga-translator-script)"}
    try:
        resp = requests.get(page_url, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [!] Failed to fetch chapter page: {e}", file=sys.stderr)
        return []

    soup = BeautifulSoup(resp.content, 'html.parser')
    img_tags = soup.find_all('img')

    image_urls = []
    for img in img_tags:
        src = img.get('src')
        if not src:
            continue
        # Check for common image file extensions
        if any(src.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
            # Convert relative URLs to absolute URLs
            full_url = urljoin(page_url, src)
            if full_url not in image_urls:
                image_urls.append(full_url)
    return sorted(image_urls)


# ----------------------------------------------------------------------
# Step 2: detect text regions (boxes only - no recognition yet)
# ----------------------------------------------------------------------
def detect_and_recognize(image: Image.Image, engine,
                         min_conf: float = 0.35,
                         min_area_frac=0.0002, max_area_frac=0.15):
    """
    Single-pass OCR: runs RapidOCR's det+rec pipeline on the full page and
    returns (kept_boxes, recognized_texts). This mirrors the original
    detect -> merge -> recognize flow, but reuses the full-page recognition
    results instead of re-running OCR on each cropped box (re-running on
    crops loses accuracy because the recognizer re-scales small crops).

    - Filters low-confidence rows and obviously-wrong regions (near-empty
      slivers, or huge regions that are almost certainly the detector getting
      confused by manga art).
    - Merges nearby boxes so one speech bubble maps to one text region.
    - For each merged region, keeps the highest-confidence recognized line.
    """
    result = engine(np.array(image))
    img_w, img_h = image.size
    img_area = img_w * img_h

    rows = []  # (rect, text, score)
    for box, text, score in _iter_ocr_results(result):
        candidate = text.strip()
        if not candidate or score < min_conf:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        area_frac = (w * h) / img_area
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue  # too tiny (noise) or too huge (bad detection)
        rows.append(((int(x1), int(y1), int(x2), int(y2)), candidate, score))

    rects = [r for r, _text, _score in rows]
    # merge_boxes returns [] for 0 or 1 boxes, so only merge when there's
    # more than one region (otherwise a single-text page would be dropped).
    boxes = merge_boxes(rects) if len(rects) > 1 else rects

    pad = 4
    kept_boxes = []
    recognized_texts = []
    for (mx1, my1, mx2, my2) in boxes:
        best = None
        for (x1, y1, x2, y2), text, score in rows:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # center of the OCR'd line
            if mx1 <= cx <= mx2 and my1 <= cy <= my2:
                if best is None or score > best[0]:
                    best = (score, text)
        if best is None:
            continue
        kept_boxes.append((max(0, mx1 - pad), max(0, my1 - pad),
                           min(img_w, mx2 + pad), min(img_h, my2 + pad)))
        recognized_texts.append(best[1])

    return kept_boxes, recognized_texts


def merge_boxes(boxes, proximity_thresh=20):
    """
    Merges overlapping or nearby bounding boxes. This is crucial for cases
    where a single speech bubble is incorrectly detected as multiple
    separate text regions. It works iteratively until no more merges can be made.
    """
    if len(boxes) <= 1:
        return []

    # Sort by top-left corner (y, then x) to process in a predictable order
    boxes.sort(key=lambda b: (b[1], b[0]))

    while True:
        merged_in_pass = False
        next_pass_boxes = []
        used_mask = [False] * len(boxes)

        for i in range(len(boxes)):
            if used_mask[i]:
                continue

            current_box = list(boxes[i])

            for j in range(i + 1, len(boxes)):
                if used_mask[j]:
                    continue

                other_box = boxes[j]
                # Check for proximity
                if (current_box[0] - proximity_thresh < other_box[2] and
                    current_box[2] + proximity_thresh > other_box[0] and
                    current_box[1] - proximity_thresh < other_box[3] and
                    current_box[3] + proximity_thresh > other_box[1]):

                    current_box[0] = min(current_box[0], other_box[0])
                    current_box[1] = min(current_box[1], other_box[1])
                    current_box[2] = max(current_box[2], other_box[2])
                    current_box[3] = max(current_box[3], other_box[3])
                    used_mask[j] = True
                    merged_in_pass = True

            next_pass_boxes.append(tuple(current_box))

        boxes = next_pass_boxes
        if not merged_in_pass:
            return boxes


# ----------------------------------------------------------------------
# Step 3 & 4: recognize text within fixed boxes (bubbles detector mode)
# ----------------------------------------------------------------------
def recognize_crop(crop: Image.Image, engine, min_conf: float = 0.35) -> str:
    """
    Runs RapidOCR on a single cropped region and returns the highest-
    confidence text line found inside it.
    """
    result = engine(np.array(crop))
    best = None
    for _box, text, score in _iter_ocr_results(result):
        candidate = text.strip()
        if not candidate or score < min_conf:
            continue
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best else ""


def recognize_boxes(image: Image.Image, boxes, engine, min_conf: float = 0.35):
    """
    Recognizes the text inside each (x1, y1, x2, y2) box by cropping and
    re-running OCR on the crop. Used when boxes come from the bubble
    detector (the full-page detection mode uses detect_and_recognize
    instead, which reuses RapidOCR's own recognition results).
    Returns (kept_boxes, recognized_texts) with boxes expanded by a small
    padding to ensure full coverage when redrawing.
    """
    pad = 4
    img_w, img_h = image.size
    kept_boxes = []
    recognized_texts = []
    for (x1, y1, x2, y2) in boxes:
        crop = image.crop((max(0, x1 - pad), max(0, y1 - pad),
                           min(img_w, x2 + pad), min(img_h, y2 + pad)))
        text = recognize_crop(crop, engine, min_conf=min_conf)
        if not text:
            continue
        expanded_box = (max(0, x1 - pad), max(0, y1 - pad),
                        min(img_w, x2 + pad), min(img_h, y2 + pad))
        kept_boxes.append(expanded_box)
        recognized_texts.append(text)
    return kept_boxes, recognized_texts


# ----------------------------------------------------------------------
# Step 5: translate
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
# Step 6: redraw the page with translated text in place of the original
# ----------------------------------------------------------------------
def pick_font_size(box, text, font_path=None, max_size=41, min_size=8):
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    size = max_size
    while size > min_size:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default(size=size)
        # Estimate wrap width: for typical fonts, character width is ~0.5-0.6 of font size.
        # For vertical text boxes, we need a more generous wrapping width.
        # A character's width is roughly half its height (size).
        wrap_width_chars = int(box_w / (size * 0.5))
        wrapped = textwrap.wrap(text, width=max(2, wrap_width_chars))

        line_h = size * 1.15
        total_h = line_h * max(1, len(wrapped))
        if total_h <= box_h + 10:
            return font, wrapped, size
        size -= 2
    font = ImageFont.truetype(font_path, min_size) if font_path else ImageFont.load_default(size=min_size)
    wrapped = textwrap.wrap(text, width=max(2, int(box_w / (min_size * 0.5))))
    return font, wrapped, min_size


def redraw_page(image: Image.Image, boxes, translations, font_path=None):
    out = image.copy()
    draw = ImageDraw.Draw(out)

    for box, translated in zip(boxes, translations):
        if not translated.strip():
            continue
        x1, y1, x2, y2 = box

        draw.rectangle([x1 - 4, y1 - 4, x2 + 4, y2 + 4], fill="white")

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
def translate_manga_page(image_source, detector: str = "easyocr", target: str = "en",
                          font_path=None, engine: str = "manga-ocr",
                          page_num=None, total_pages=None):
    """
    Translates a manga page image. The OCR backend is RapidOCR (ONNX), which
    replaced the old torch-based easyocr/manga-ocr stack so the serverless
    bundle stays under Vercel's size limit.
    """
    page_prefix = ""
    if page_num is not None and total_pages is not None:
        page_prefix = f"[Page {page_num}/{total_pages}] "

    print(f"{page_prefix}[1/6] Loading image...")
    if isinstance(image_source, str):  # It's a URL
        image = load_image_from_url(image_source)
    elif isinstance(image_source, bytes):  # It's raw image data
        image = Image.open(io.BytesIO(image_source)).convert("RGB")
    elif isinstance(image_source, Image.Image):  # Already a PIL image
        image = image_source
    else:
        raise TypeError("image_source must be a URL (str), image data (bytes), or PIL Image object")

    MAX_DIM = 1600
    if max(image.size) > MAX_DIM:
        ratio = MAX_DIM / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    ocr_engine = get_ocr_engine()
    if detector == "bubbles":
        print(f"{page_prefix}[2/6] Detecting text regions (detector=bubbles)")
        boxes = detect_bubbles(image)
        print(f"{page_prefix}      -> Found {len(boxes)} bubble region(s).")
        print(f"{page_prefix}[3/6] Recognizing text in bubbles (backend=rapidocr)")
        kept_boxes, recognized_texts = recognize_boxes(image, boxes, ocr_engine)
        print(f"{page_prefix}      -> Recognized {len(recognized_texts)} line(s).")
    else:
        print(f"{page_prefix}[2/6] Detecting and recognizing text (backend=rapidocr, single pass)")
        kept_boxes, recognized_texts = detect_and_recognize(image, ocr_engine)
        print(f"{page_prefix}      -> Recognized {len(recognized_texts)} line(s).")
        boxes = kept_boxes

    print(f"{page_prefix}[4/6] Translating {len(recognized_texts)} line(s) as a single batch "
          f"(more reliable than one request per line)")
    translations = translate_all(recognized_texts, target=target)
    for original, translated in zip(recognized_texts, translations):
        print(f"{page_prefix}      JP: {original!r:35s}  ->  {target.upper()}: {translated!r}")

    print(f"{page_prefix}[5/6] Rendering translated page...")
    result_img = redraw_page(image, kept_boxes, translations, font_path=font_path)
    print(f"{page_prefix}[6/6] Done.")
    del image  # release the decoded PIL image now that we're done with it
    return kept_boxes, recognized_texts, translations, result_img