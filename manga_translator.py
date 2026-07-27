"""
Manga Translator
-----------------
Give it an image URL (a manga page), and it will:
  1. Download the image
  2. Detect Japanese text regions on the page (OCR)
  3. Translate each detected line into English (or any target language)
  4. Paint over the original Japanese text and draw the translated text
     back into the same spot on the page
  5. Save the result as a new image, plus print a text-only report

Usage:
    python manga_translator.py <image_url> [--target en] [--out result.png]

Example:
    python manga_translator.py https://example.com/onepiece_ch1_p3.png
"""

import argparse
import io
import sys
import textwrap

import requests
from PIL import Image, ImageDraw, ImageFont
import easyocr
from deep_translator import GoogleTranslator


# ----------------------------------------------------------------------
# Step 1: fetch the image
# ----------------------------------------------------------------------
def load_image_from_url(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0 (manga-translator-script)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return img


# ----------------------------------------------------------------------
# Step 2: detect + recognize Japanese text
# ----------------------------------------------------------------------
def detect_text(image: Image.Image, reader: easyocr.Reader):
    """
    Returns a list of (box, text, confidence)
    box = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]  (4 corner points)
    """
    import numpy as np
    np_img = np.array(image)
    results = reader.readtext(np_img, detail=1, paragraph=False)
    return results


# ----------------------------------------------------------------------
# Step 3: translate
# ----------------------------------------------------------------------
def translate_text(text: str, target: str = "en", source: str = "ja") -> str:
    if not text.strip():
        return ""
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:
        print(f"  [!] translation failed for '{text}': {e}", file=sys.stderr)
        return text


# ----------------------------------------------------------------------
# Step 4: redraw the page with translated text in place of the original
# ----------------------------------------------------------------------
def box_to_rect(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def pick_font_size(rect, text, font_path=None, max_size=40, min_size=10):
    x1, y1, x2, y2 = rect
    box_w, box_h = x2 - x1, y2 - y1
    size = max_size
    while size > min_size:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        wrapped = textwrap.wrap(text, width=max(1, int(box_w / (size * 0.55))))
        line_h = size * 1.15
        total_h = line_h * max(1, len(wrapped))
        if total_h <= box_h + 10:  # small tolerance
            return font, wrapped, size
        size -= 2
    font = ImageFont.truetype(font_path, min_size) if font_path else ImageFont.load_default()
    wrapped = textwrap.wrap(text, width=max(1, int(box_w / (min_size * 0.55))))
    return font, wrapped, min_size


def redraw_page(image: Image.Image, detections, translations, font_path=None):
    out = image.copy()
    draw = ImageDraw.Draw(out)

    for (box, _orig_text, _conf), translated in zip(detections, translations):
        rect = box_to_rect(box)
        x1, y1, x2, y2 = rect

        # Cover the original Japanese text with a white box
        draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], fill="white")

        if not translated.strip():
            continue

        font, wrapped_lines, size = pick_font_size(rect, translated, font_path)
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
                          font_path=None, langs=("ja", "en")):
    print(f"[1/4] Downloading image from: {url}")
    image = load_image_from_url(url)

    print(f"[2/4] Running OCR (languages={langs}) ... this loads models on first run")
    reader = easyocr.Reader(list(langs), gpu=False)
    detections = detect_text(image, reader)
    print(f"      -> Found {len(detections)} text region(s)")

    print(f"[3/4] Translating detected text to '{target}'")
    translations = []
    for box, text, conf in detections:
        translated = translate_text(text, target=target)
        translations.append(translated)
        print(f"      JP: {text!r:40s} (conf={conf:.2f})  ->  {target.upper()}: {translated!r}")

    print(f"[4/4] Rendering translated page -> {out_path}")
    result_img = redraw_page(image, detections, translations, font_path=font_path)
    result_img.save(out_path)
    print("Done.")
    return detections, translations, result_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate a manga page from an image URL.")
    parser.add_argument("url", help="URL of the manga page image")
    parser.add_argument("--target", default="en", help="Target language code (default: en)")
    parser.add_argument("--out", default="translated.png", help="Output image path")
    parser.add_argument("--font", default=None, help="Path to a .ttf font file (recommended for non-English targets)")
    args = parser.parse_args()

    translate_manga_page(args.url, target=args.target, out_path=args.out, font_path=args.font)
