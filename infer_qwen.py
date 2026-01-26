import os
import io
import base64
import json
import re
from PIL import Image
from openai import OpenAI

# ==========================================
# 1. Configuration
# ==========================================
# Use "qwen-vl-plus" or "qwen-vl-max" for Qwen-VL on Bailian
MODEL_NAME = "qwen-vl-max"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Set your API Key
api_key = "sk-d4fa9916adb34e23894306858fc25f0a"
if not api_key:
    raise ValueError("Please set the DASHSCOPE_API_KEY environment variable.")

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL,
)

INPUT_FILE = "/data/feihong/ckpt/test.txt"
OUTPUT_FILE = "./runs/infer_results.json"
SATELLITE_DIR = "/data/feihong/image_1024"
API_TIMEOUT = 60

results = {}

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        results = json.load(f)
    skipped = len(results)
    print(f"Resuming: {skipped} images already processed")


# ==========================================
# 2. Image Processing & Encoding Function
# ==========================================
def process_and_encode_image(image_path, target_size=(1024, 1024)):
    """
    Opens an image, converts to RGB, resizes (preserving aspect ratio),
    center crops to target_size, and returns the Base64 string.
    """
    try:
        with Image.open(image_path) as img:
            # 1. Ensure image is in RGB mode (handles PNG transparency etc.)
            img = img.convert("RGB")

            # 2. Calculate aspect-preserving resize dimensions
            # We resize so the *smallest* dimension becomes 1024
            width, height = img.size
            target_w, target_h = target_size
            aspect_ratio = width / height
            target_aspect = target_w / target_h

            if aspect_ratio > target_aspect:
                # Image is wider than target: resize based on height
                new_height = target_h
                new_width = int(target_h * aspect_ratio)
            else:
                # Image is taller than target: resize based on width
                new_width = target_w
                new_height = int(target_w / aspect_ratio)

            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 3. Center Crop to exactly 1024x1024
            left = (new_width - target_w) / 2
            top = (new_height - target_h) / 2
            right = (new_width + target_w) / 2
            bottom = (new_height + target_h) / 2

            img_cropped = img_resized.crop((left, top, right, bottom))

            # 4. Convert to Base64
            buffered = io.BytesIO()
            img_cropped.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return img_str

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def parse_bbox(text):
    """
    Parse bbox from API response. Handles both JSON and text formats.

    Args:
        text: API response text

    Returns:
        List of 4 integers [x1, y1, x2, y2] or None if parsing fails
    """
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and "bbox_2d" in data[0]:
                return data[0]["bbox_2d"]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    nums = re.findall(r"\d+", text)
    if len(nums) >= 4:
        return [int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])]

    return None


with open(INPUT_FILE, "r") as f:
    image_paths = [line.strip() for line in f if line.strip()]

print(f"Found {len(image_paths)} image pairs to process")

for idx, drone_path in enumerate(image_paths):
    name = os.path.basename(os.path.dirname(drone_path))
    img_name = name

    if name in results:
        print(f"[{idx + 1}/{len(image_paths)}] Skipping {name} (already processed)")
        continue

    satellite_path = os.path.join(SATELLITE_DIR, f"{name}.png")

    if not os.path.exists(satellite_path):
        satellite_path = satellite_path.replace(".png", ".jpeg")

    if not os.path.exists(drone_path) or not os.path.exists(satellite_path):
        print(f"Skipping {name}: files not found")
        continue

    base64_img1 = process_and_encode_image(drone_path, target_size=(512, 512))
    base64_img2 = process_and_encode_image(satellite_path)

    if not base64_img1 or not base64_img2:
        print(f"Skipping {name}: failed to process images")
        continue

    text_prompt = """Locate the the drone image region in the satellite image.
Output format: [x1, y1, x2, y2] range from 0 to 1024
IMPORTANT: Output ONLY the bbox numbers in brackets, no JSON, no extra text.
"""

    try:
        print(f"[{idx + 1}/{len(image_paths)}] Processing {name}...")
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img1}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img2}"
                            },
                        },
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ],
            temperature=0.01,
            top_p=0.1,
            timeout=API_TIMEOUT,
        )

        generated_text = completion.choices[0].message.content
        print(f"  Response: {generated_text}")

        bbox_list = parse_bbox(generated_text)
        if not bbox_list:
            print(f"  Warning: No bbox found in response")
            continue

        x1, y1, x2, y2 = bbox_list
        img_width, img_height = 3840, 2160

        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))

        pixel_bbox = [x1, y1, x2, y2]
        results[img_name] = pixel_bbox
        print(f"  Bbox: {pixel_bbox}")

    except openai.APITimeoutError as e:
        print(f"  Timeout, saving partial results...")
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        raise e
    except Exception as e:
        print(f"Error processing {name}: {e}")
        continue

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! Results saved to {OUTPUT_FILE}")
print(f"Total processed: {len(results)}/{len(image_paths)}")
