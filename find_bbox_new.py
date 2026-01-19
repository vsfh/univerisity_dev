import os
import torch
import json
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from loguru import logger
from tqdm import tqdm
import cv2
import numpy as np

try:
    from perceptron.tensorstream import VisionType
    from perceptron.tensorstream.ops import tensor_stream_token_view, modality_mask
    from perceptron.pointing.parser import extract_points
except ImportError:
    raise ImportError("perceptron package required")

HF_PATH = "PerceptronAI/Isaac-0.1"
CACHE_DIR = "/data/feihong/hf_cache"
INPUT_DIR = "/data/feihong/image_1024"
OUTPUT_DIR = "/data/feihong/univerisity_dev/runs/img"
OUTPUT_JSON = "bbox_results_new.json"
NUM_IMAGES = 100
PROMPT = "Locate and provide the bounding box for the single building positioned at the center of this aerial image. Focus on the main central structure only."

os.environ["HF_HOME"] = CACHE_DIR
logger.info(f"Loading processor and config from HF checkpoint: {HF_PATH}")
config = AutoConfig.from_pretrained(
    HF_PATH, trust_remote_code=True, cache_dir=CACHE_DIR
)
tokenizer = AutoTokenizer.from_pretrained(
    HF_PATH, trust_remote_code=True, use_fast=False, cache_dir=CACHE_DIR
)
processor = AutoProcessor.from_pretrained(
    HF_PATH, trust_remote_code=True, cache_dir=CACHE_DIR
)
processor.tokenizer = tokenizer

logger.info(f"Loading AutoModelForCausalLM from HF checkpoint: {HF_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    HF_PATH, trust_remote_code=True, cache_dir=CACHE_DIR
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = model.to(device=device, dtype=dtype)
model.eval()

logger.info(f"Model loaded on {device} with dtype {dtype}")


def document_to_messages(document, vision_token="<image>"):
    messages, images = [], []
    for item in document:
        if item.get("type") == "text" and item.get("content"):
            messages.append(
                {"role": item.get("role", "user"), "content": item["content"]}
            )
        elif item.get("type") == "image" and item.get("content"):
            img = Image.open(item["content"]).convert("RGB")
            images.append(img)
            messages.append({"role": item.get("role", "user"), "content": vision_token})
    return messages, images


def decode_tensor_stream(tensor_stream, tokenizer):
    token_view = tensor_stream_token_view(tensor_stream)
    mod = modality_mask(tensor_stream)
    text_tokens = token_view[(mod != VisionType.image.value)]
    return tokenizer.decode(
        text_tokens[0] if len(text_tokens.shape) > 1 else text_tokens
    )


def visualize_bbox(image, bbox, output_path):
    img_with_bbox = image.copy()
    draw = ImageDraw.Draw(img_with_bbox)

    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

    img_with_bbox.save(output_path, "PNG")
    logger.info(f"Saved visualization to {output_path}")


def fine_tune_bbox():
    """
    Interactive bounding box fine-tuning using OpenCV.
    Load bbox results, display images with bboxes, allow dragging to adjust.
    Save changes to bbox_changed_res.json.
    Press 'q' to quit, 'Enter' to save and proceed to next image.
    """
    INPUT_JSON = "/data/feihong/univerisity_dev/bbox_results_new.json"
    OUTPUT_JSON = "/data/feihong/univerisity_dev/bbox_changed_res.json"
    IMAGE_DIR = "/data/feihong/image_1024"
    DISPLAY_SIZE = 640
    NUM_IMAGES = 50

    with open(INPUT_JSON, "r") as f:
        bbox_data = json.load(f)

    image_names = sorted([k for k in bbox_data.keys()])[:NUM_IMAGES]
    if not image_names:
        logger.error("No images found in bbox results")
        return

    changed_results = {}

    current_idx = 0
    current_image_name = None
    original_image = None
    scaled_image = None
    current_bbox_scaled = None
    original_bbox_scaled = None
    scale_factor = 1.0
    pad_x = 0
    pad_y = 0

    dragging = False
    drag_start = None
    bbox_start = None
    handle_size = 8
    resize_edge = None

    WINDOW_NAME = "BBox Fine-tuning - Press 'q' to quit, 'Enter' to save & next, 'n' next, 'p' prev, 'r' reset"

    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging, drag_start, bbox_start, current_bbox_scaled, resize_edge

        if current_bbox_scaled is None:
            return

        x1, y1, x2, y2 = current_bbox_scaled
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if event == cv2.EVENT_LBUTTONDOWN:
            margin = handle_size + 5

            if x1 - margin <= x <= x1 + margin and y1 - margin <= y <= y1 + margin:
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "tl"
            elif x2 - margin <= x <= x2 + margin and y2 - margin <= y <= y2 + margin:
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "br"
            elif x2 - margin <= x <= x2 + margin and y1 - margin <= y <= y1 + margin:
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "tr"
            elif x1 - margin <= x <= x1 + margin and y2 - margin <= y <= y2 + margin:
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "bl"
            elif x1 - margin <= x <= x2 + margin and y1 <= y <= y1 + margin:
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "t"
            elif x1 - margin <= x <= x2 + margin and y2 - margin <= y <= y2 + margin:
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "b"
            elif x1 <= x <= x1 + margin and y1 - margin <= y <= y2 + margin:
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "l"
            elif x2 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin:
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "r"
            elif x1 <= x <= x2 and y1 <= y <= y2:
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging and drag_start is not None and bbox_start is not None:
                dx = x - drag_start[0]
                dy = y - drag_start[1]

                x1_s, y1_s, x2_s, y2_s = bbox_start

                if resize_edge in ("tl", "tr", "t", "l"):
                    if resize_edge in ("tl", "l"):
                        x1_s = max(0, min(x1_s + dx, x2_s - 1))
                    if resize_edge in ("tl", "t"):
                        y1_s = max(0, min(y1_s + dy, y2_s - 1))

                if resize_edge in ("br", "bl", "b", "r"):
                    if resize_edge in ("br", "r"):
                        x2_s = min(DISPLAY_SIZE, max(x2_s + dx, x1_s + 1))
                    if resize_edge in ("br", "bl", "b"):
                        y2_s = min(DISPLAY_SIZE, max(y2_s + dy, y1_s + 1))

                if resize_edge == "tr":
                    x2_s = min(DISPLAY_SIZE, max(x2_s + dx, x1_s + 1))
                    y1_s = max(0, min(y1_s + dy, y2_s - 1))

                if resize_edge == "bl":
                    x1_s = max(0, min(x1_s + dx, x2_s - 1))
                    y2_s = min(DISPLAY_SIZE, max(y2_s + dy, y1_s + 1))

                if resize_edge is None:
                    width = x2_s - x1_s
                    height = y2_s - y1_s
                    x1_s = max(0, min(x1_s + dx, DISPLAY_SIZE - width))
                    y1_s = max(0, min(y1_s + dy, DISPLAY_SIZE - height))
                    x2_s = x1_s + width
                    y2_s = y1_s + height

                current_bbox_scaled = [x1_s, y1_s, x2_s, y2_s]
                draw_image()

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            drag_start = None
            bbox_start = None
            resize_edge = None

    def apply_letterbox(image, target_size):
        """Apply letterbox resizing to target_size while maintaining aspect ratio."""
        orig_w, orig_h = image.size
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        img_resized = image.resize((new_w, new_h), Image.LANCZOS)

        padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        padded.paste(img_resized, (paste_x, paste_y))

        return padded, scale, paste_x, paste_y

    def convert_bbox_to_scaled(bbox, scale, pad_x, pad_y):
        """Convert original image coordinates to scaled display coordinates."""
        x1 = int(bbox[0] * scale) + pad_x
        y1 = int(bbox[1] * scale) + pad_y
        x2 = int(bbox[2] * scale) + pad_x
        y2 = int(bbox[3] * scale) + pad_y
        return [x1, y1, x2, y2]

    def convert_bbox_to_original(bbox_scaled, scale, pad_x, pad_y):
        """Convert scaled display coordinates back to original image coordinates."""
        x1 = max(0, int((bbox_scaled[0] - pad_x) / scale))
        y1 = max(0, int((bbox_scaled[1] - pad_y) / scale))
        x2 = max(0, int((bbox_scaled[2] - pad_x) / scale))
        y2 = max(0, int((bbox_scaled[3] - pad_y) / scale))
        return [x1, y1, x2, y2]

    def draw_image():
        """Draw the current image with bbox and handles."""
        if scaled_image is None or current_bbox_scaled is None:
            return

        img_array = np.array(scaled_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        x1, y1, x2, y2 = [int(c) for c in current_bbox_scaled]

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.circle(img_bgr, (x1, y1), handle_size, (0, 0, 255), -1)
        cv2.circle(img_bgr, (x2, y2), handle_size, (0, 0, 255), -1)
        cv2.circle(img_bgr, (x1, y2), handle_size, (0, 0, 255), -1)
        cv2.circle(img_bgr, (x2, y1), handle_size, (0, 0, 255), -1)

        cv2.putText(
            img_bgr,
            f"Image: {current_image_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_bgr,
            f"Coords: [{x1}, {y1}, {x2}, {y2}]",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_bgr,
            f"Press: Enter=Save&Next, n=Next, p=Prev, r=Reset, q=Quit",
            (10, DISPLAY_SIZE - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow(WINDOW_NAME, img_bgr)

    def load_image(idx):
        """Load image at given index and set up bbox."""
        nonlocal current_idx, current_image_name, original_image, scaled_image
        nonlocal current_bbox_scaled, original_bbox_scaled, scale_factor, pad_x, pad_y

        current_idx = idx
        current_image_name = image_names[current_idx]

        img_path = os.path.join(IMAGE_DIR, current_image_name)
        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            return False

        original_image = Image.open(img_path).convert("RGB")
        scaled_image, scale_factor, pad_x, pad_y = apply_letterbox(
            original_image, DISPLAY_SIZE
        )

        original_bbox = bbox_data[current_image_name]
        current_bbox_scaled = convert_bbox_to_scaled(
            original_bbox, scale_factor, pad_x, pad_y
        )
        original_bbox_scaled = list(current_bbox_scaled)

        draw_image()
        return True

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    logger.info(f"Starting bbox fine-tuning for {len(image_names)} images")
    logger.info("Controls: Enter=Save&Next, n=Next, p=Prev, r=Reset, q=Quit")

    load_image(0)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            if changed_results:
                with open(OUTPUT_JSON, "w") as f:
                    json.dump(changed_results, f, indent=2)
                logger.info(
                    f"Saved {len(changed_results)} changed results to {OUTPUT_JSON}"
                )
            break

        elif key == 13 or key == ord("\n") or key == ord("s"):
            bbox_original = convert_bbox_to_original(
                current_bbox_scaled, scale_factor, pad_x, pad_y
            )
            changed_results[current_image_name] = bbox_original
            logger.info(f"Saved bbox for {current_image_name}: {bbox_original}")

            if current_idx < len(image_names) - 1:
                load_image(current_idx + 1)
            else:
                logger.info("Reached last image")
                load_image(current_idx)

        elif key == ord("n") or key == 2555904:
            if current_idx < len(image_names) - 1:
                load_image(current_idx + 1)

        elif key == ord("p") or key == 2424832:
            if current_idx > 0:
                load_image(current_idx - 1)

        elif key == ord("r"):
            current_bbox_scaled = list(original_bbox_scaled)
            draw_image()

    cv2.destroyAllWindows()

    if changed_results:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(changed_results, f, indent=2)
        logger.info(
            f"Fine-tuning complete. Saved {len(changed_results)} changed results to {OUTPUT_JSON}"
        )
    else:
        logger.info("Fine-tuning complete. No changes made.")


def generate_bbox_response(image_path):
    document = [
        {"type": "text", "content": "<hint>BOX</hint>", "role": "user"},
        {"type": "image", "content": image_path, "role": "user"},
        {"type": "text", "content": PROMPT, "role": "user"},
    ]

    messages, images = document_to_messages(document, vision_token=config.vision_token)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=text, images=images, return_tensors="pt")
    tensor_stream = inputs["tensor_stream"].to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            tensor_stream=tensor_stream,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        generated_text = processor.tokenizer.decode(
            generated_ids[0], skip_special_tokens=False
        )

    return generated_text, images[0] if images else None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    for i in tqdm(range(NUM_IMAGES)):
        img_name = f"{i:04d}.png"
        img_path = os.path.join(INPUT_DIR, img_name)

        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            continue

        logger.info(f"Processing {img_name}")
        generated_text, image = generate_bbox_response(img_path)

        boxes = extract_points(generated_text, expected="box")
        if not boxes:
            logger.info(f"No building detected in {img_name}, skipping")
            continue

        box = boxes[0]
        img_width, img_height = image.size

        norm_x1, norm_y1 = box.top_left.x, box.top_left.y
        norm_x2, norm_y2 = box.bottom_right.x, box.bottom_right.y

        x1 = int((norm_x1 / 1000.0) * img_width)
        y1 = int((norm_y1 / 1000.0) * img_height)
        x2 = int((norm_x2 / 1000.0) * img_width)
        y2 = int((norm_y2 / 1000.0) * img_height)

        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))

        pixel_bbox = [x1, y1, x2, y2]
        results[img_name] = pixel_bbox

        # vis_path = os.path.join(OUTPUT_DIR, f"{img_name.replace('.png', '')}_bbox.png")
        # visualize_bbox(image, pixel_bbox, vis_path)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Complete. Processed {len(results)} images with detected buildings.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--finetune":
        fine_tune_bbox()
    else:
        main()
