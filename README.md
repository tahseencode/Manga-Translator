# Manga Translator

Translates Japanese manga pages into English. Paste a chapter URL or upload a page, and it finds the speech bubbles, reads the Japanese, translates it, and draws the English back in.

**Demo:** [manga-translator-production-52f3.up.railway.app](https://manga-translator-production-52f3.up.railway.app)

## How it works

Bubbles are found with OpenCV contour detection, the text is read with RapidOCR, translated through `deep-translator` (Google Translate), and redrawn with Pillow in a comic font.

Each page is its own request (`/translate_single`) instead of one long stream. It keeps memory low and doesn't break on flaky mobile networks.

## Stack

Flask, Gunicorn, OpenCV, RapidOCR (ONNX, no PyTorch), deep-translator, Pillow, BeautifulSoup

## Endpoints

- `GET /` serves the UI
- `POST /get_pages` takes a chapter URL and returns its page image URLs
- `POST /translate_single` translates one page
- `POST /translate` translates an uploaded image or direct image URL

## Run locally

```bash
git clone https://github.com/tahseencode/Manga-Translator.git
cd Manga-Translator
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`. Needs Python 3.12.

## Deployment

Runs from the included Dockerfile on Railway. The `vercel.json` is left over from an earlier attempt, since Vercel's size limits couldn't fit the OCR dependencies.

## Known issues

- Oddly shaped or shaded bubbles can get missed
- Google Translate struggles with slang and context-heavy lines
- Some mobile carriers still cause connection problems

Made by [Tahseen](https://github.com/tahseencode)
