# Made by Tahseen
import base64
import gc
import io
import os
import sys
import json
from flask import Flask, request, render_template, jsonify
from manga_translator import translate_manga_page, find_image_urls_on_page

app = Flask(__name__, template_folder='.')

script_dir = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(script_dir, "animeace2.ttf")
if not os.path.exists(FONT_PATH):
    print("[Warning] animeace2.ttf not found, falling back to default font.")
    FONT_PATH = None


@app.route('/')
def index():
    return render_template('index.html')


def _encode_image_and_data(boxes, original_texts, translations, result_img):
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return {
        "translated_image": "data:image/png;base64," + img_str,
        "translation_data": [{"box": box, "original": orig, "translation": trans}
                              for box, orig, trans in zip(boxes, original_texts, translations)],
    }


@app.route('/get_pages', methods=['POST'])
def get_pages_endpoint():
    """Given a chapter URL, returns the list of individual page image URLs."""
    source_url = request.form.get('image_url')
    if not source_url:
        return jsonify({"error": "No URL provided."}), 400
    try:
        image_urls = find_image_urls_on_page(source_url)
        if not image_urls:
            return jsonify({"error": "No images found on that page."}), 400
        return jsonify({"image_urls": image_urls})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/translate_single', methods=['POST'])
def translate_single_endpoint():
    """Translates exactly one page image URL and returns a plain JSON response
    (no streaming). Called once per page from the frontend so each request
    fully completes and releases memory before the next one starts — this
    avoids the OOM kill that happened when all pages were processed inside
    one long-lived SSE request, and also avoids mobile carrier proxies
    breaking long-lived streaming connections."""
    data = request.get_json()
    page_url = data.get('image_url')
    page_num = data.get('page_num')
    total_pages = data.get('total_pages')

    if not page_url:
        return jsonify({"error": "No image_url provided."}), 400

    try:
        boxes, original_texts, translations, result_img = translate_manga_page(
            image_source=page_url, font_path=FONT_PATH,
            page_num=page_num, total_pages=total_pages
        )
        result = _encode_image_and_data(boxes, original_texts, translations, result_img)
        del boxes, original_texts, translations, result_img
        gc.collect()
        return jsonify(result)
    except Exception as e:
        print(f"  [!] FAILED to translate page {page_num}: {e}", file=sys.stderr)
        gc.collect()
        return jsonify({"error": f"Failed to translate page. Reason: {str(e)}"}), 500


@app.route('/translate', methods=['POST'])
def translate_single_upload_endpoint():
    """Handles a direct file upload or a single direct image URL (not a
    chapter page to scrape). Kept as plain JSON — no streaming needed since
    it's only ever one image."""
    image_file = request.files.get('image_file')
    uploaded_bytes = image_file.read() if image_file and image_file.filename != '' else None
    source_url = request.form.get('image_url')

    try:
        if uploaded_bytes:
            image_source = uploaded_bytes
        elif source_url:
            image_source = source_url
        else:
            return jsonify({"error": "No image file or URL provided."}), 400

        boxes, original_texts, translations, result_img = translate_manga_page(
            image_source=image_source, font_path=FONT_PATH
        )
        result = {
            "page_index": 0,
            "source_image_url": source_url if source_url else None,
        }
        result.update(_encode_image_and_data(boxes, original_texts, translations, result_img))
        del boxes, original_texts, translations, result_img
        gc.collect()
        return jsonify(result)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        gc.collect()
        return jsonify({"error": str(e)}), 500