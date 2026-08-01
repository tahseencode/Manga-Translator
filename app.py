import base64
import io
import os
import sys
import json
from flask import Flask, request, render_template, Response
from manga_translator import translate_manga_page, initialize_ocr_models, find_image_urls_on_page

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

@app.route('/translate', methods=['POST'])
def translate_endpoint():
    # Extract data from the request context immediately.
    # This makes the generator independent of the request context and prevents
    # "Working outside of request context" errors with streaming.
    image_file = request.files.get('image_file')
    source_url_from_form = request.form.get('image_url')

    def generate_stream(uploaded_file, source_url):
        try:
            # Case 1: File Upload
            if uploaded_file and uploaded_file.filename != '':
                image_source = uploaded_file.read()
                yield f"event: init\ndata: {json.dumps({'total_pages': 1})}\n\n"
                
                boxes, original_texts, translations, result_img = translate_manga_page(image_source=image_source, font_path=FONT_PATH)
                buffered = io.BytesIO()
                result_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                page_result = {
                    "page_index": 0,
                    "translated_image": "data:image/png;base64," + img_str,
                    "translation_data": [{"box": box, "original": orig, "translation": trans} for box, orig, trans in zip(boxes, original_texts, translations)],
                    "source_image_url": None # No source URL for file uploads
                }
                yield f"event: page_result\ndata: {json.dumps(page_result)}\n\n"

            # Case 2: URL provided
            elif source_url:
                try:
                    # First, try to treat it as a single direct image URL
                    boxes, original_texts, translations, result_img = translate_manga_page(image_source=source_url, font_path=FONT_PATH)
                    yield f"event: init\ndata: {json.dumps({'total_pages': 1})}\n\n"
                    
                    buffered = io.BytesIO()
                    result_img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    page_result = {
                        "page_index": 0,
                        "translated_image": "data:image/png;base64," + img_str,
                        "translation_data": [{"box": box, "original": orig, "translation": trans} for box, orig, trans in zip(boxes, original_texts, translations)],
                        "source_image_url": source_url
                    }
                    yield f"event: page_result\ndata: {json.dumps(page_result)}\n\n"

                except ValueError as e:
                    # If that fails, assume it's a chapter page and stream it
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
                                buffered = io.BytesIO()
                                result_img.save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()
                                
                                page_result.update({
                                    "translated_image": "data:image/png;base64," + img_str,
                                    "translation_data": [{"box": box, "original": orig, "translation": trans} for box, orig, trans in zip(boxes, original_texts, translations)],
                                })
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
    return Response(generate_stream(image_file, source_url_from_form), mimetype='text/event-stream')

# This part runs when the module is imported.
# It pre-loads the heavy models for better performance on serverless platforms like Vercel.
initialize_ocr_models()