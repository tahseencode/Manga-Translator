# Manga Translator

A Flask web app that translates Japanese manga pages into English — automatically. Give it a chapter URL or upload an image, and it detects the speech bubbles, runs OCR on the Japanese text, translates it, and redraws the English translation back onto the page.

**Live demo:** [manga-translator-production-52f3.up.railway.app](https://manga-translator-production-52f3.up.railway.app)

## How it works

1. **Fetch** — pulls page images either from a chapter URL (scraped with BeautifulSoup) or a direct upload/image URL.
2. **Detect** — locates speech bubbles on the page using contour detection (`bubble_detector.py`).
3. **OCR** — reads the Japanese text inside each bubble with [RapidOCR](https://github.com/RapidAI/RapidOCR) (ONNX-based, PP-OCRv6 models, no PyTorch dependency).
4. **Translate** — sends the recognized text through `deep-translator` (Google Translate backend) from Japanese to English.
5. **Redraw** — clears each bubble and redraws it with the translated text, auto-sized to fit, using a bundled comic font (`animeace2.ttf`).

Each manga page is processed and returned as its own request/response (`/translate_single`) rather than one long streaming connection. This keeps memory usage low per request (avoiding out-of-memory kills on constrained hosting) and holds up better on unstable mobile networks, where long-lived streaming connections tend to get cut off by carrier proxies.

## Tech stack

- **Backend:** Flask, Gunicorn
- **OCR:** RapidOCR (ONNX runtime, no PyTorch — keeps deploy size small)
- **Bubble detection:** OpenCV (contour-based)
- **Translation:** `deep-translator`
- **Image processing:** Pillow, NumPy
- **Scraping:** `requests` + BeautifulSoup4
- **Frontend:** single-page `index.html` (custom UI)

## API endpoints

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the web UI |
| `/get_pages` | POST | Given a chapter URL, scrapes and returns the list of page image URLs |
| `/translate_single` | POST | Translates one page image (used per-page by the frontend) |
| `/translate` | POST | Translates a single uploaded image or direct image URL |

## Running locally

```bash
git clone https://github.com/tahseencode/Manga-Translator.git
cd Manga-Translator
pip install -r requirements.txt
python app.py
```

The app will be available at `http://localhost:5000` (or via Gunicorn: `gunicorn -b 0.0.0.0:7860 app:app`).

### Requirements

- Python 3.12
- See [`requirements.txt`](requirements.txt) for full dependency list

## Deployment

Includes a `Dockerfile` (installs `libgl1`/`libglib2.0-0` for OpenCV, then runs Gunicorn) and is currently deployed on [Railway](https://railway.app). A `vercel.json` is also included from an earlier deployment attempt, though Vercel's serverless function size limits made Railway/Docker a better fit for the OCR dependencies.

## Known limitations

- Bubble detection uses simple contour/whiteness heuristics, so unusually shaped or shaded bubbles may be missed.
- Translation quality depends on Google Translate via `deep-translator`, so idiomatic or context-heavy dialogue may translate awkwardly.
- Mobile connectivity on some carrier networks can still be inconsistent; a custom domain is the suggested long-term fix.

## Credits

Made by [Tahseen](https://github.com/tahseencode).
