# floor-plan-tool

A local Flask app for extracting room geometry from 2D floor plan images using an LM Studio vision model.

## Features

- Upload a 2D floor plan image
- Extract rooms with normalized coordinates and dimensions
- Render a generated 2D layout
- Optional spatial extraction for rooms, doors, and relations
- Export the rendered plan as PNG or PDF

## Requirements

- Python 3.10+
- LM Studio running locally with an OpenAI-compatible endpoint
- A vision-capable model loaded in LM Studio

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
LOCAL_MODEL_URL=http://127.0.0.1:8000/v1/chat/completions
LOCAL_MODEL_NAME=qwen/qwen3.5-9b
LOCAL_API_KEY=lm-studio
```

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.
