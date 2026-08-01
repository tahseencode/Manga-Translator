


import argparse
import io
import os
import sys
import textwrap
import time
from urllib.parse import urljoin
import concurrent.futures

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup
import easyocr
from deep_translator import GoogleTranslator

from bubble_detector import detect_bubbles
# manga_ocr is optional - the script still works with plain easyocr if it's
# not installed, just with lower recognition quality on manga-style text.
try:
    from manga_ocr import MangaOcr
    _HAS_MANGA_OCR = True
except ImportError:
    _HAS_MANGA_OCR = False

_mocr_singleton = None
_easyocr_singleton = None

def get_manga_ocr():
    global _mocr_singleton
    if _mocr_singleton is None:
        print("      loading manga-ocr model (first call only)...")
        _mocr_singleton = MangaOcr()
    return _mocr_singleton

def get_easyocr_reader():
    global _easyocr_singleton
    if _easyocr_singleton is None:
        print("      loading easyocr models (first call only)...")
        _easyocr_singleton = easyocr.Reader(["ja", "en"], gpu=False)
    return _easyocr_singleton

def initialize_ocr_models():
    get_easyocr_reader()
    if _HAS_MANGA_OCR:
        get_manga_ocr()

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
def pick_font_size(box, text, font_path=None, max_size=46, min_size=9):
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
    page_prefix = ""
    if page_num is not None and total_pages is not None:
        page_prefix = f"[Page {page_num}/{total_pages}] "

    if engine == "manga-ocr" and not _HAS_MANGA_OCR:
        print(f"{page_prefix}      [!] manga-ocr not installed, falling back to easyocr recognizer.", file=sys.stderr)
        engine = "easyocr"

    print(f"{page_prefix}[1/6] Loading image...")
    if isinstance(image_source, str): # It's a URL
        image = load_image_from_url(image_source)
    elif isinstance(image_source, bytes): # It's raw image data
        image = Image.open(io.BytesIO(image_source)).convert("RGB")
    elif isinstance(image_source, Image.Image): # Already a PIL image
        image = image_source
    else:
        raise TypeError("image_source must be a URL (str), image data (bytes), or PIL Image object")

    reader = get_easyocr_reader()
    print(f"{page_prefix}[2/6] Detecting text regions (detector={detector})")
    if detector == "bubbles":
        boxes = detect_bubbles(image)
    else: # easyocr
        boxes = detect_boxes(image, reader)
    
    print(f"{page_prefix}      -> Found {len(boxes)} initial regions, merging...")
    boxes = merge_boxes(boxes)
    print(f"{page_prefix}      -> Down to {len(boxes)} final region(s) after merging.")
    print(f"{page_prefix}[3/6] Recognizing text (this step is placeholder, see next)")

    print(f"{page_prefix}[4/6] Recognizing text (engine={engine})")
    kept_boxes = []
    recognized_texts = []
    for box in boxes:
        text = recognize_box(image, box, reader, engine=engine)
        if not text:
            continue
        # Expand box slightly to ensure full coverage when redrawing
        x1, y1, x2, y2 = box
        pad = 4
        expanded_box = (max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad)
        kept_boxes.append(expanded_box)
        recognized_texts.append(text)

    print(f"{page_prefix}[5/6] Translating {len(recognized_texts)} line(s) as a single batch "
          f"(more reliable than one request per line)")
    translations = translate_all(recognized_texts, target=target)
    for original, translated in zip(recognized_texts, translations):
        print(f"{page_prefix}      JP: {original!r:35s}  ->  {target.upper()}: {translated!r}")

    print(f"{page_prefix}[6/6] Rendering translated page...")
    result_img = redraw_page(image, kept_boxes, translations, font_path=font_path)
    print(f"{page_prefix}Done.")
    # Return the boxes and text for the interactive preview
    return kept_boxes, recognized_texts, translations, result_img