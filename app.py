import os
import json
import re
import base64
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)

# LM Studio local OpenAI-compatible vision endpoint.
LOCAL_MODEL_URL = os.environ.get(
    "LOCAL_MODEL_URL", "http://127.0.0.1:8000/v1/chat/completions"
)
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "qwen/qwen3.5-9b")
LOCAL_API_KEY = os.environ.get("LOCAL_API_KEY", "lm-studio")

SYSTEM_PROMPT = """You are a floor plan digitizer. Analyze the floor plan image carefully.
Return ONLY a valid JSON array with every room.

Example format:
[{"name":"Kitchen","type":"kitchen","x_norm":0.15,"y_norm":0.55,"w_norm":0.20,"h_norm":0.16,"width_ft":8.0,"length_ft":8.5}]

Rules:
- x_norm, y_norm = top-left corner as fraction of image (0.0 to 1.0)
- w_norm, h_norm = room width/height as fraction of image
- width_ft/length_ft = read from printed labels on the drawing
- Classification: "CB" or "Cupboard" -> "storage", "T & B" or "Toilet" -> "bathroom".
- All rooms must be non-overlapping and placed exactly where they appear in the image
- Include ALL rooms: bedrooms, bathrooms, kitchen, living, dining, balconies, utility, passages, storage
- Do NOT include doors, entry arcs, or compass symbols as rooms
- Return ONLY the JSON array, no explanation, no thinking"""

PROMPT_TEMPLATE = "The image size is WIDTH={w}px and HEIGHT={h}px. Extract every room with normalized coordinates (x_norm=x_px/WIDTH, y_norm=y_px/HEIGHT, etc). Return ONLY a JSON array."


ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


def build_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LOCAL_API_KEY}",
    }


def detect_mime_from_b64(b64_str):
    """Detect image MIME type from the first bytes of the base64 data."""
    try:
        header = base64.b64decode(b64_str[:32])
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            return 'image/png'
        if header[:2] == b'\xff\xd8':
            return 'image/jpeg'
        if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            return 'image/webp'
        if header[:3] == b'GIF':
            return 'image/gif'
    except Exception:
        pass
    return None


def convert_to_png_b64(b64_str):
    """Convert any image format to PNG base64 for LM Studio compatibility."""
    try:
        from PIL import Image
        from io import BytesIO
        
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_bytes))
        
        # Convert to RGB if necessary (e.g., RGBA, P mode)
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')
        
        buf = BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8'), 'image/png'
    except ImportError:
        print("[convert] Pillow not installed, skipping conversion")
        return b64_str, 'image/png'
    except Exception as e:
        print(f"[convert] Image conversion failed: {e}")
        return b64_str, 'image/png'


def build_multimodal_content(prompt, base64_img=None, mime_type="image/jpeg"):
    content = [{"type": "text", "text": prompt}]
    if base64_img:
        # Strip invisible characters that break base64 validation in LM Studio
        clean_b64 = base64_img.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Auto-detect MIME type from actual image bytes
        detected_mime = detect_mime_from_b64(clean_b64)
        if detected_mime:
            mime_type = detected_mime
        
        # LM Studio only accepts PNG and JPEG — convert WebP/GIF/others to PNG
        if mime_type not in ('image/png', 'image/jpeg'):
            print(f"[payload] Converting {mime_type} to PNG for LM Studio compatibility")
            clean_b64, mime_type = convert_to_png_b64(clean_b64)
            
        image_value = f"data:{mime_type};base64,{clean_b64}"
            
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_value},
            }
        )
    return content


def build_payload(prompt, base64_img=None, mime_type="image/jpeg"):
    content = build_multimodal_content(prompt, base64_img, mime_type)
    return {
        "model": LOCAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "/no_think\n" + SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        "temperature": 0.1,
        "max_tokens": 16384,
    }


def extract_json_text(content):
    if not isinstance(content, str):
        raise ValueError("Model response content was not text")

    cleaned = content.strip()

    # Strip <think>...</think> blocks (or truncated <think> with no closing tag)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    elif cleaned.startswith("<think>"):
        # Model used all tokens on thinking, no JSON produced
        cleaned = ""

    # Strip GLM special tokens like <|begin_of_box|>, <|end_of_box|>, etc.
    cleaned = re.sub(r'<\|[^|]+\|>', '', cleaned).strip()

    # Strip markdown code fences
    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

    # Try to find a JSON array
    match = ARRAY_PATTERN.search(cleaned)
    if match:
        return match.group(0)

    # If nothing found, return whatever we have
    return cleaned


def extract_json_object_text(content):
    """Extract the outer JSON object from model text content."""
    if not isinstance(content, str):
        raise ValueError("Model response content was not text")

    cleaned = content.strip()

    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    elif cleaned.startswith("<think>"):
        cleaned = ""

    cleaned = re.sub(r'<\|[^|]+\|>', '', cleaned).strip()

    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start:end + 1]
    return cleaned


def parse_dimension_string(dim_str):
    """Parse strings like '11\'2" x 11\'6"' or '4.32m x 8.03m' into (width_mm, length_mm)."""
    if not dim_str:
        return 3000, 3000
    
    dim_str = dim_str.lower().replace(' ', '')
    parts = dim_str.split('x')
    if len(parts) != 2:
        return 3000, 3000
        
    def parse_side(side):
        # Handle feet and inches (e.g. 11'2")
        ft_match = re.search(r'(\d+(?:\.\d+)?)\'', side)
        in_match = re.search(r'(\d+(?:\.\d+)?)(?:"|in)', side)
        
        if ft_match or in_match:
            ft = float(ft_match.group(1)) if ft_match else 0
            inches = float(in_match.group(1)) if in_match else 0
            return round((ft * 304.8) + (inches * 25.4))
            
        # Handle meters (e.g. 4.32m)
        m_match = re.search(r'(\d+(?:\.\d+)?)m', side)
        if m_match:
            return round(float(m_match.group(1)) * 1000)
            
        # Handle raw mm
        mm_match = re.search(r'(\d+(?:\.\d+)?)mm', side)
        if mm_match:
            return round(float(mm_match.group(1)))
            
        # Fallback to pure numbers assuming feet
        num_match = re.search(r'(\d+(?:\.\d+)?)', side)
        if num_match:
            return round(float(num_match.group(1)) * 304.8)
            
        return 3000

    return parse_side(parts[0]), parse_side(parts[1])

def parse_response(resp_json, img_w=800, img_h=600):
    try:
        content = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"[parse] Unexpected response structure: {json.dumps(resp_json)[:500]}")
        raise ValueError(f"Could not parse model response: {e}") from e

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        content = "\n".join(text_parts)

    # Log raw content for debugging
    print(f"[parse] Raw model content ({len(content)} chars): {content[:300]}")

    json_text = extract_json_text(content)
    print(f"[parse] Extracted JSON text: {json_text[:300]}")
    
    if not json_text or json_text.isspace():
        print("[parse] Model returned empty content, returning empty rooms list")
        return []

    rooms = repair_and_parse_json(json_text)

    FT_TO_MM = 304.8
    for room in rooms:
        # --- Dimensions ---
        if "dimensions" in room:
            w, l = parse_dimension_string(room["dimensions"])
            room.setdefault("width_mm", w)
            room.setdefault("length_mm", l)
        if room.get("width_ft") and "width_mm" not in room:
            room["width_mm"] = round(float(room["width_ft"]) * FT_TO_MM)
        if room.get("length_ft") and "length_mm" not in room:
            room["length_mm"] = round(float(room["length_ft"]) * FT_TO_MM)
        room.setdefault("width_mm", 3000)
        room.setdefault("length_mm", 3000)
        
        # --- Spatial position (normalized 0-1) ---
        # If model gave x_norm/y_norm directly, use them
        if "x_norm" in room and "y_norm" in room:
            pass  # already normalized
        # If model gave pixel coords, convert to normalized
        elif "x_pixel" in room and "y_pixel" in room:
            room["x_norm"] = round(room["x_pixel"] / img_w, 4) if img_w else 0
            room["y_norm"] = round(room["y_pixel"] / img_h, 4) if img_h else 0
        else:
            room["x_norm"] = 0
            room["y_norm"] = 0
            
        # Normalized size
        if "w_norm" not in room:
            if "width_px" in room:
                room["w_norm"] = round(room["width_px"] / img_w, 4) if img_w else 0.2
            else:
                room["w_norm"] = 0.2
        if "h_norm" not in room:
            if "height_px" in room:
                room["h_norm"] = round(room["height_px"] / img_h, 4) if img_h else 0.2
            else:
                room["h_norm"] = 0.2
    
    return rooms


def repair_and_parse_json(text):
    """Try multiple strategies to parse potentially malformed JSON from the model."""

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Fix common issues
    fixed = text.strip()
    # Remove trailing commas before ] or }
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    # If truncated (no closing ]), add it
    if fixed.startswith('[') and not fixed.rstrip().endswith(']'):
        # Find the last complete object (ending with })
        last_brace = fixed.rfind('}')
        if last_brace > 0:
            fixed = fixed[:last_brace + 1] + ']'
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Extract individual room objects with regex
    print("[parse] Attempting regex extraction of room objects")
    room_pattern = re.compile(
        r'\{[^{}]*?"name"\s*:\s*"[^"]*?"[^{}]*?"type"\s*:\s*"[^"]*?"[^{}]*?\}',
        re.DOTALL
    )
    matches = room_pattern.findall(text)
    rooms = []
    for m in matches:
        try:
            room = json.loads(m)
            rooms.append(room)
        except json.JSONDecodeError:
            # Try to fix this individual object
            try:
                fixed_m = re.sub(r',\s*}', '}', m)
                room = json.loads(fixed_m)
                rooms.append(room)
            except json.JSONDecodeError:
                print(f"[parse] Could not parse room object: {m[:100]}")

    if rooms:
        print(f"[parse] Regex extracted {len(rooms)} rooms")
        return rooms

    print(f"[parse] All parse strategies failed. Raw text: {text[:500]}")
    raise ValueError(f"Could not parse model JSON output")


SPATIAL_SYSTEM_PROMPT = """You are a floor-plan spatial digitizer.
Return ONLY one valid JSON object with this exact top-level schema:
{
  "rooms": [
    {
      "name": "ROOM NAME",
      "type": "bedroom|living|kitchen|bathroom|dining|corridor|balcony|study|utility|storage|default",
      "x_norm": 0.0,
      "y_norm": 0.0,
      "w_norm": 0.0,
      "h_norm": 0.0,
      "dimensions": "raw printed dimension text if visible, else null",
      "width_mm": 0,
      "length_mm": 0,
      "source": "printed|estimated",
      "confidence": "high|medium|low"
    }
  ],
  "doors": [
    {
      "id": "D1",
      "x_norm": 0.0,
      "y_norm": 0.0,
      "w_norm": 0.0,
      "h_norm": 0.0,
      "swing": "left|right|double|unknown",
      "connects": ["Room A", "Room B or exterior"],
      "confidence": "high|medium|low"
    }
  ],
  "spatial_relations": [
    {
      "subject": "Door D1",
      "relation": "on_wall_of|between|north_of|south_of|east_of|west_of|adjacent_to",
      "object": "Room name or area",
      "confidence": "high|medium|low"
    }
  ],
  "notes": ["optional short notes"]
}

Rules:
- Coordinates must be normalized 0.0..1.0 relative to image size.
- Include all visible rooms and all visible doors.
- Use printed dimensions when visible. If not visible, estimate width_mm and length_mm and set source="estimated".
- Return ONLY JSON object, no markdown, no prose, no code fences, no think text.
"""


def _clamp01(value, default=0.0):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, num))


def _bbox_from_norm(x_norm, y_norm, w_norm, h_norm):
    x1 = _clamp01(x_norm, 0.0)
    y1 = _clamp01(y_norm, 0.0)
    w = max(0.0, min(1.0, float(w_norm) if isinstance(w_norm, (int, float, str)) else 0.0))
    h = max(0.0, min(1.0, float(h_norm) if isinstance(h_norm, (int, float, str)) else 0.0))
    x2 = _clamp01(x1 + w, x1)
    y2 = _clamp01(y1 + h, y1)
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def _rect_intersection_area(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _point_to_rect_distance(px, py, rect):
    rx1, ry1, rx2, ry2 = rect
    dx = max(rx1 - px, 0.0, px - rx2)
    dy = max(ry1 - py, 0.0, py - ry2)
    return (dx * dx + dy * dy) ** 0.5


def _infer_door_connections(doors, rooms):
    """Infer which rooms each door connects based on overlap and proximity."""
    room_refs = []
    for room in rooms:
        if not isinstance(room, dict):
            continue
        room_name = room.get("name") or "Unknown Room"
        bbox = room.get("bbox_norm")
        if isinstance(bbox, list) and len(bbox) == 4:
            room_refs.append((room_name, bbox))

    if not room_refs:
        return

    for i, door in enumerate(doors):
        if not isinstance(door, dict):
            continue

        db = door.get("bbox_norm")
        if not (isinstance(db, list) and len(db) == 4):
            continue

        door_area = max((db[2] - db[0]) * (db[3] - db[1]), 1e-9)
        cx = (db[0] + db[2]) / 2.0
        cy = (db[1] + db[3]) / 2.0

        scored = []
        for room_name, rb in room_refs:
            inter = _rect_intersection_area(db, rb)
            overlap_ratio = inter / door_area
            dist = _point_to_rect_distance(cx, cy, rb)
            scored.append((room_name, overlap_ratio, dist))

        scored.sort(key=lambda x: (-x[1], x[2]))

        picked = []
        for room_name, overlap_ratio, dist in scored:
            if overlap_ratio > 0.01 or dist < 0.06:
                picked.append(room_name)
            if len(picked) == 2:
                break

        if not picked:
            picked = [scored[0][0]]

        if len(picked) == 1:
            picked.append("exterior")

        door.setdefault("id", f"D{i + 1}")
        door["connects"] = picked[:2]


def parse_spatial_response(resp_json, img_w=800, img_h=600):
    try:
        content = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"[spatial-parse] Unexpected response structure: {json.dumps(resp_json)[:500]}")
        raise ValueError(f"Could not parse model response: {e}") from e

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        content = "\n".join(text_parts)

    json_text = extract_json_object_text(content)
    if not json_text or json_text.isspace():
        return {"rooms": [], "doors": [], "spatial_relations": [], "notes": ["empty model output"]}

    parsed = repair_and_parse_json(json_text)
    if isinstance(parsed, list):
        # Fallback when model returns only room array.
        parsed = {"rooms": parsed, "doors": [], "spatial_relations": [], "notes": ["model returned array only"]}
    if not isinstance(parsed, dict):
        raise ValueError("Spatial response JSON is not an object")

    rooms = parsed.get("rooms", [])
    doors = parsed.get("doors", [])
    relations = parsed.get("spatial_relations", [])
    notes = parsed.get("notes", [])

    FT_TO_MM = 304.8
    for room in rooms:
        if not isinstance(room, dict):
            continue

        if "dimensions" in room and room["dimensions"]:
            w, l = parse_dimension_string(room["dimensions"])
            room.setdefault("width_mm", w)
            room.setdefault("length_mm", l)

        if room.get("width_ft") and "width_mm" not in room:
            room["width_mm"] = round(float(room["width_ft"]) * FT_TO_MM)
        if room.get("length_ft") and "length_mm" not in room:
            room["length_mm"] = round(float(room["length_ft"]) * FT_TO_MM)

        room.setdefault("width_mm", 3000)
        room.setdefault("length_mm", 3000)

        if "x_norm" not in room and "x_pixel" in room:
            room["x_norm"] = round(float(room["x_pixel"]) / img_w, 4) if img_w else 0
        if "y_norm" not in room and "y_pixel" in room:
            room["y_norm"] = round(float(room["y_pixel"]) / img_h, 4) if img_h else 0
        if "w_norm" not in room and "width_px" in room:
            room["w_norm"] = round(float(room["width_px"]) / img_w, 4) if img_w else 0.2
        if "h_norm" not in room and "height_px" in room:
            room["h_norm"] = round(float(room["height_px"]) / img_h, 4) if img_h else 0.2

        room.setdefault("x_norm", 0)
        room.setdefault("y_norm", 0)
        room.setdefault("w_norm", 0.2)
        room.setdefault("h_norm", 0.2)
        room.setdefault("source", "estimated")
        room.setdefault("confidence", "medium")

        room["bbox_norm"] = _bbox_from_norm(
            room.get("x_norm", 0),
            room.get("y_norm", 0),
            room.get("w_norm", 0.2),
            room.get("h_norm", 0.2),
        )
        room["center_norm"] = [
            round((room["bbox_norm"][0] + room["bbox_norm"][2]) / 2.0, 4),
            round((room["bbox_norm"][1] + room["bbox_norm"][3]) / 2.0, 4),
        ]

    for door in doors:
        if not isinstance(door, dict):
            continue
        if "x_norm" not in door and "x_pixel" in door:
            door["x_norm"] = round(float(door["x_pixel"]) / img_w, 4) if img_w else 0
        if "y_norm" not in door and "y_pixel" in door:
            door["y_norm"] = round(float(door["y_pixel"]) / img_h, 4) if img_h else 0
        if "w_norm" not in door and "width_px" in door:
            door["w_norm"] = round(float(door["width_px"]) / img_w, 4) if img_w else 0.05
        if "h_norm" not in door and "height_px" in door:
            door["h_norm"] = round(float(door["height_px"]) / img_h, 4) if img_h else 0.05
        door.setdefault("x_norm", 0)
        door.setdefault("y_norm", 0)
        door.setdefault("w_norm", 0.05)
        door.setdefault("h_norm", 0.05)
        door.setdefault("swing", "unknown")
        door.setdefault("connects", [])
        door.setdefault("confidence", "medium")

        door["bbox_norm"] = _bbox_from_norm(
            door.get("x_norm", 0),
            door.get("y_norm", 0),
            door.get("w_norm", 0.05),
            door.get("h_norm", 0.05),
        )
        door["center_norm"] = [
            round((door["bbox_norm"][0] + door["bbox_norm"][2]) / 2.0, 4),
            round((door["bbox_norm"][1] + door["bbox_norm"][3]) / 2.0, 4),
        ]

    _infer_door_connections(doors, rooms)

    rel_subjects = {(r.get("subject"), r.get("relation"), r.get("object")) for r in relations if isinstance(r, dict)}
    for door in doors:
        did = door.get("id", "Door")
        for room_name in (door.get("connects") or []):
            triple = (f"Door {did}", "between", room_name)
            if triple not in rel_subjects:
                relations.append(
                    {
                        "subject": triple[0],
                        "relation": triple[1],
                        "object": triple[2],
                        "confidence": door.get("confidence", "medium"),
                    }
                )
                rel_subjects.add(triple)

    if not isinstance(relations, list):
        relations = []
    if not isinstance(notes, list):
        notes = [str(notes)]

    return {
        "rooms": rooms,
        "doors": doors,
        "spatial_relations": relations,
        "notes": notes,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-status")
def model_status():
    payload = build_payload("Reply with exactly [] and nothing else.")
    try:
        resp = requests.post(
            LOCAL_MODEL_URL,
            headers=build_headers(),
            json=payload,
            timeout=20.0,
        )
        if resp.status_code >= 400:
            return jsonify(
                {
                    "ok": False,
                    "model": LOCAL_MODEL_NAME,
                    "url": LOCAL_MODEL_URL,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                }
            ), 503
        return jsonify({"ok": True, "model": LOCAL_MODEL_NAME, "url": LOCAL_MODEL_URL})
    except requests.ConnectionError:
        return jsonify(
            {
                "ok": False,
                "model": LOCAL_MODEL_NAME,
                "url": LOCAL_MODEL_URL,
                "error": "Cannot connect to LM Studio. Start the local server and load the vision model.",
            }
        ), 503
    except requests.Timeout:
        return jsonify(
            {
                "ok": False,
                "model": LOCAL_MODEL_NAME,
                "url": LOCAL_MODEL_URL,
                "error": "LM Studio connection timed out.",
            }
        ), 504

@app.route("/api/extract", methods=["POST"])
def extract_plan():
    data = request.json or {}
    base64_img = data.get("image", "")
    mime_type = data.get("mimeType", "image/jpeg")
    img_w = data.get("imgWidth", 800)
    img_h = data.get("imgHeight", 600)

    if not base64_img:
        return jsonify({"error": "No image provided"}), 400

    # Build prompt with image dimensions
    prompt = PROMPT_TEMPLATE.format(w=img_w, h=img_h)
    payload = build_payload(prompt, base64_img, mime_type)

    try:
        print(f"[extract] Calling model: {LOCAL_MODEL_NAME} at {LOCAL_MODEL_URL}")
        print(f"[extract] Image: {len(base64_img)} chars, {img_w}x{img_h}px, MIME: {mime_type}")
        
        resp = requests.post(
            LOCAL_MODEL_URL,
            headers=build_headers(),
            json=payload,
            timeout=600.0,
        )
        
        if resp.status_code >= 400:
            response_error = resp.text[:500]
            print(f"[extract] API error {resp.status_code}: {response_error}")
            return jsonify(
                {"error": f"Model server error ({resp.status_code}): {response_error[:200]}"}
            ), 500
            
        rooms = parse_response(resp.json(), img_w, img_h)

        print(f"[extract] Success: {len(rooms)} rooms extracted")
        return jsonify(rooms)
    except requests.ConnectionError:
        err_msg = f"Cannot connect to model server at {LOCAL_MODEL_URL}. Is LM Studio running?"
        print(f"[extract] {err_msg}")
        return jsonify({"error": err_msg}), 503
    except requests.Timeout:
        return jsonify(
            {
                "error": "Model server timed out. The image may be too large or the model is overloaded."
            }
        ), 504
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": f"Failed to parse model response: {e}"}), 500
    except Exception as e:
        print(f"[extract] Failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/extract-spatial", methods=["POST"])
def extract_spatial_plan():
    data = request.json or {}
    base64_img = data.get("image", "")
    mime_type = data.get("mimeType", "image/jpeg")
    img_w = data.get("imgWidth", 800)
    img_h = data.get("imgHeight", 600)

    if not base64_img:
        return jsonify({"error": "No image provided"}), 400

    prompt = (
        f"Image size: WIDTH={img_w}px HEIGHT={img_h}px. "
        "Extract rooms, doors, visible measurement labels, and spatial relations. "
        "Coordinates must be normalized in range 0..1."
    )
    payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "/no_think\n" + SPATIAL_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_img.strip().replace('\n', '').replace('\r', '').replace(' ', '')}"}},
            ]},
        ],
        "temperature": 0.1,
        "max_tokens": 16384,
    }

    try:
        print(f"[extract-spatial] Calling model: {LOCAL_MODEL_NAME} at {LOCAL_MODEL_URL}")
        resp = requests.post(
            LOCAL_MODEL_URL,
            headers=build_headers(),
            json=payload,
            timeout=600.0,
        )
        if resp.status_code >= 400:
            response_error = resp.text[:500]
            print(f"[extract-spatial] API error {resp.status_code}: {response_error}")
            return jsonify({"error": f"Model server error ({resp.status_code}): {response_error[:200]}"}), 500

        spatial = parse_spatial_response(resp.json(), img_w, img_h)
        print(
            "[extract-spatial] Success: "
            f"{len(spatial.get('rooms', []))} rooms, "
            f"{len(spatial.get('doors', []))} doors, "
            f"{len(spatial.get('spatial_relations', []))} relations"
        )
        return jsonify(spatial)
    except requests.ConnectionError:
        err_msg = f"Cannot connect to model server at {LOCAL_MODEL_URL}. Is LM Studio running?"
        print(f"[extract-spatial] {err_msg}")
        return jsonify({"error": err_msg}), 503
    except requests.Timeout:
        return jsonify({"error": "Model server timed out. The image may be too large or the model is overloaded."}), 504
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": f"Failed to parse model response: {e}"}), 500
    except Exception as e:
        print(f"[extract-spatial] Failed: {e}")
        return jsonify({"error": str(e)}), 500


SKU_SYSTEM_PROMPT = """You are a precise furniture SKU digitizer.
Extract ONLY the furniture rows visible in the image. Do NOT invent data.

For each row return:
- sku: description text used as SKU identifier
- s_no: row number
- description: exact text as printed
- size_raw: full size string as printed
- width_mm: W value in mm (null if missing)
- length_mm: L value in mm (null if missing)
- height_mm: H value in mm (null if missing)
- depth_mm: D value in mm (null if missing)
- units: quantity number
- net_price_rs: price per unit as float
- total_rs: total price as float
- source: "printed"
- confidence: "high"

Return ONLY a valid JSON array. No markdown, no explanation."""


@app.route("/api/extract-skus", methods=["POST"])
def extract_skus():
    data = request.get_json(force=True)
    img_b64 = data.get("image_b64", "")
    img_mime = data.get("mime_type", "image/jpeg")

    if not img_b64:
        return jsonify({"error": "No image provided"}), 400

    payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SKU_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{img_mime};base64,{img_b64}"}},
                {"type": "text", "text": "Extract all furniture SKU rows from this catalogue image. Return ONLY the JSON array. /no_think"}
            ]}
        ],
        "max_tokens": 8192,
        "temperature": 0.0,
        "stream": False
    }

    try:
        resp = requests.post(LOCAL_MODEL_URL, headers=build_headers(), json=payload, timeout=120)
        raw = resp.json()["choices"][0]["message"]["content"]
        # Strip <think> blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        match = ARRAY_PATTERN.search(raw)
        if not match:
            return jsonify({"error": "No JSON array found in response", "raw": raw[:500]}), 500
        skus = repair_and_parse_json(match.group())
        print(f"[extract-skus] Success: {len(skus)} SKUs extracted")
        return jsonify(skus)
    except Exception as e:
        print(f"[extract-skus] Failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
