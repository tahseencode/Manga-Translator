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
from bubble_detector import detect_bubbles


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
def _get_text_dimensions(text_line, font):
    # Pango backend (used on Linux) and Raqm backend (used on Mac/Windows)
    # behave differently. Use textbbox to get a consistent result.
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text_line)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        return font.getsize(text_line)

def _wrap_text_and_find_font(text, width, height, font_path, max_size, min_size):
    """
    Finds the best font size and wraps text to fit a box.
    This is a more complex implementation that avoids using textwrap directly
    and tries to find a visually pleasing line break.
    """
    words = text.split()
    
    for size in range(max_size, min_size - 1, -1):
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        line_height = sum(_get_text_dimensions("A", font)) * 0.9  # Approx height

        lines = []
        current_line = ""
        
        if not words:
            continue

        current_line = words[0]
        for word in words[1:]:
            # Check if adding the new word exceeds the width
            if _get_text_dimensions(current_line + " " + word, font)[0] <= width:
                current_line += " " + word
            else:
                # If it exceeds, push the current line and start a new one
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        
        total_height = len(lines) * line_height
        
        if total_height <= height:
            max_line_width = 0
            for l in lines:
                max_line_width = max(max_line_width, _get_text_dimensions(l, font)[0])

            if max_line_width <= width:
                 return font, lines, size

    # If no size fits, return the smallest font and best-effort wrap
    font = ImageFont.truetype(font_path, min_size) if font_path else ImageFont.load_default()
    line_height = sum(_get_text_dimensions("A", font)) * 0.9
    lines = []
    current_line = words[0]
    for word in words[1:]:
        if _get_text_dimensions(current_line + " " + word, font)[0] <= width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)

    return font, lines, min_size


def redraw_page(image: Image.Image, boxes, translations, font_path=None):
    """
    Erases the original text bubbles and draws new, clean bubbles with the
    translated text.
    """
    out = image.copy()
    draw = ImageDraw.Draw(out)

    for box, translated in zip(boxes, translations):
        if not translated.strip():
            continue
        
        # Erase the old bubble by drawing a white box over it
        x1, y1, x2, y2 = box
        # Use a slightly larger box to ensure full erasure of bubble tail etc.
        draw.rectangle([x1 - 10, y1 - 10, x2 + 10, y2 + 10], fill="white", width=0)

        # Determine the size needed for the translated text
        # Use a slightly smaller box to get the font size, to leave padding
        font, wrapped_lines, size = _wrap_text_and_find_font(
            translated, (x2 - x1) * 0.9, (y2 - y1) * 0.9, font_path, max_size=32, min_size=8
        )
        full_text = "\n".join(line.strip() for line in wrapped_lines)
        if not full_text:
            continue

        # Get the dimensions of the wrapped text block
        try:
            text_bbox = draw.multiline_textbbox((0, 0), full_text, font=font, align="center", spacing=4)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_w, text_h = 0, 0
            line_height = sum(_get_text_dimensions("A", font)) * 0.9
            for line in wrapped_lines:
                lw, _ = _get_text_dimensions(line.strip(), font)
                text_w = max(text_w, lw)
            text_h = len(wrapped_lines) * line_height


        # Create a new bubble that fits the text
        bubble_w = text_w + 40  # Add horizontal padding
        bubble_h = text_h + 30  # Add vertical padding
        
        # Center the new bubble where the old one was
        center_x = x1 + (x2 - x1) / 2
        center_y = y1 + (y2 - y1) / 2
        
        new_bubble_x1 = center_x - bubble_w / 2
        new_bubble_y1 = center_y - bubble_h / 2
        new_bubble_x2 = center_x + bubble_w / 2
        new_bubble_y2 = center_y + bubble_h / 2
        
        # Draw the new bubble (white fill, black outline)
        draw.ellipse(
            [new_bubble_x1, new_bubble_y1, new_bubble_x2, new_bubble_y2],
            fill="white",
            outline="black",
            width=2
        )

        # Draw the text in the new bubble
        try:
            draw.multiline_text((center_x, center_y), full_text, fill="black", font=font,
                                align="center", anchor="mm", spacing=4)
        except TypeError:
            text_x = center_x - text_w / 2
            text_y = center_y - text_h / 2
            draw.multiline_text((text_x, text_y), full_text, fill="black", font=font, 
                                align="center", spacing=4)

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

    print("[3/6] Detecting speech bubbles")
    boxes = detect_bubbles(image)
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
    parser = argparse.ArgumentParser(
        description="Translates a manga page from a direct image URL. Example: \n"
                    "  python manga_translator.py \"http://example.com/manga_page.jpg\"",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("url", help="URL of the manga page image")
    parser.add_argument("--target", default="en", help="Target language code (default: en)")
    parser.add_argument("--out", default="translated.png", help="Output image path")
    parser.add_argument("--font", default=None, help="Path to a .ttf font file (recommended for non-English targets)")
    parser.add_argument("--engine", default="manga-ocr", choices=["manga-ocr", "easyocr"],
                         help="Recognition engine (manga-ocr is much more accurate on manga text)")
    args = parser.parse_args()

    translate_manga_page(args.url, target=args.target, out_path=args.out,
                          font_path=args.font, engine=args.engine)