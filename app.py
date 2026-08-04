# Made by Tahseen
import base64
import io
import os
import sys
import json
from flask import Flask, request, render_template, Response, stream_with_context
from manga_translator import translate_manga_page, find_image_urls_on_page

# Let Flask handle the static folder automatically.
# By default, it will serve files from a 'static' subfolder
# at the '/static' URL path.
app = Flask(__name__, template_folder='.')

# --- Configuration ---
# Looks for 'animeace2.ttf' in the same directory as the script.
script_dir = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(script_dir, "animeace2.ttf")
if not os.path.exists(FONT_PATH):
    print("[Warning] animeace2.ttf not found, falling back to default font.")
    FONT_PATH = None

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

def _encode_image_and_data(boxes, original_texts, translations, result_img):
    """Encodes the translated image to base64 and formats the text data."""
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "translated_image": "data:image/png;base64," + img_str,
        "translation_data": [{"box": box, "original": orig, "translation": trans} for box, orig, trans in zip(boxes, original_texts, translations)],
    }


@app.route('/translate', methods=['POST'])
def translate_endpoint():
    # Read all uploaded data BEFORE the request handler returns. The
    # generator below is evaluated lazily by Flask while streaming the
    # response, by which point the request's file handles are already
    # closed - so reading the file inside the generator would raise
    # "I/O operation on closed file".
    image_file = request.files.get('image_file')
    uploaded_bytes = image_file.read() if image_file and image_file.filename != '' else None
    source_url_from_form = request.form.get('image_url')

    def generate_stream(payload_bytes, source_url):
        try:
            # Case 1: File Upload
            if payload_bytes:
                image_source = payload_bytes
                yield f"event: init\ndata: {json.dumps({'total_pages': 1})}\n\n"
                
                boxes, original_texts, translations, result_img = translate_manga_page(image_source=image_source, font_path=FONT_PATH)
                page_result = {
                    "page_index": 0,
                    "source_image_url": None # No source URL for file uploads
                }
                page_result.update(_encode_image_and_data(boxes, original_texts, translations, result_img))
                yield f"event: page_result\ndata: {json.dumps(page_result)}\n\n"

            # Case 2: URL provided
            elif source_url:
                try:
                    # First, try to treat it as a single direct image URL
                    yield f"event: init\ndata: {json.dumps({'total_pages': 1})}\n\n"
                    boxes, original_texts, translations, result_img = translate_manga_page(image_source=source_url, font_path=FONT_PATH)
                    page_result = {
                        "page_index": 0,
                        "source_image_url": source_url
                    }
                    page_result.update(_encode_image_and_data(boxes, original_texts, translations, result_img))
                    yield f"event: page_result\ndata: {json.dumps(page_result)}\n\n"

                except ValueError as e:
                    # If loading the URL as an image fails, assume it's a chapter page and scrape it.
                    if "direct image link" in str(e):
                        print(f"URL is not a direct image link. Attempting to scrape and stream: {source_url}")
                        image_urls = find_image_urls_on_page(source_url)
                        if not image_urls:
                            raise Exception("The URL was not a direct image link, and no images could be found on the page.")
                        
                        yield f"event: init\ndata: {json.dumps({'total_pages': len(image_urls)})}\n\n"

                        for i, page_url in enumerate(image_urls):
                            page_result = {"page_index": i, "source_image_url": page_url}
                            try:
                                print(f"\n--- Translating Page {i+1}/{len(image_urls)} ---")
                                boxes, original_texts, translations, result_img = translate_manga_page(
                                    image_source=page_url, font_path=FONT_PATH, page_num=i+1, total_pages=len(image_urls)
                                )
                                success_data = _encode_image_and_data(boxes, original_texts, translations, result_img)
                                page_result.update(success_data)
                            except Exception as page_e:
                                print(f"  [!] FAILED to translate page {i+1}: {page_e}", file=sys.stderr)
                                page_result["error"] = f"Failed to translate page. Reason: {str(page_e)}"
                            
                            yield f"event: page_result\ndata: {json.dumps(page_result)}\n\n"
                    else:
                        raise e # Re-raise other ValueErrors
            else:
                raise Exception("No image file or URL provided.")

            yield "event: finished\ndata: {}\n\n"

        except Exception as e:
            print(f"An error occurred during stream generation: {e}", file=sys.stderr)
            error_data = {"error": str(e)}
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    # Pass the extracted data into the generator when creating the response.
    return Response(stream_with_context(generate_stream(uploaded_bytes, source_url_from_form)), mimetype='text/event-stream')
