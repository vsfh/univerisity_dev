import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
from loguru import logger

INPUT_TXT = "test.txt"
OUTPUT_JSON = "output.json"
DRONE_VIEW_FOLDER = r"E:\intern\drone_view"
SATELLITE_FOLDER = r"E:\intern\image_1024"
INITIAL_BBOX = [1536, 656, 2268, 1374]
DISPLAY_SIZE = 1280
HANDLE_SIZE = 10


def load_paths_from_txt(txt_path):
    """Load paths from text.txt and extract image keys."""
    paths = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(line)
    keys = []
    for path in paths:
        parts = path.split("/")
        key = parts[-2]
        keys.append(key)
    return list(set(keys)), paths


def load_existing_json(json_path):
    """Load existing JSON or return empty dict."""
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} existing entries from {json_path}")
        return data
    return {}


def save_json(json_path, data):
    """Save data to JSON file."""
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(data)} entries to {json_path}")


def get_image_paths(key):
    """Get drone and satellite image paths for a given key."""
    drone_path = os.path.join(DRONE_VIEW_FOLDER, "gallery_drone", key, "image-01.jpeg")
    satellite_path = os.path.join(SATELLITE_FOLDER, f"{key}.png")
    return drone_path, satellite_path


def scale_bbox(bbox, scale):
    """Scale bbox by a factor."""
    return [coord * scale for coord in bbox]


def apply_letterbox(image, target_size):
    """Apply letterbox resizing maintaining aspect ratio."""
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
    """Convert original bbox to scaled display coordinates."""
    x1 = int(bbox[0] * scale) + pad_x
    y1 = int(bbox[1] * scale) + pad_y
    x2 = int(bbox[2] * scale) + pad_x
    y2 = int(bbox[3] * scale) + pad_y
    return [x1, y1, x2, y2]


def convert_bbox_to_original(bbox_scaled, scale, pad_x, pad_y):
    """Convert scaled bbox back to original coordinates."""
    x1 = max(0, int((bbox_scaled[0] - pad_x) / scale))
    y1 = max(0, int((bbox_scaled[1] - pad_y) / scale))
    x2 = max(0, int((bbox_scaled[2] - pad_x) / scale))
    y2 = max(0, int((bbox_scaled[3] - pad_y) / scale))
    return [x1, y1, x2, y2]


def draw_combined_image(drone_img, satellite_img_scaled, bbox_scaled, key):
    """Draw drone and satellite images side by side with bbox on satellite."""
    drone_array = np.array(drone_img)
    satellite_array = np.array(satellite_img_scaled)

    drone_bgr = cv2.cvtColor(drone_array, cv2.COLOR_RGB2BGR)
    satellite_bgr = cv2.cvtColor(satellite_array, cv2.COLOR_RGB2BGR)

    drone_h, drone_w = drone_bgr.shape[:2]
    sat_h, sat_w = satellite_bgr.shape[:2]

    combined_h = max(drone_h, sat_h)
    combined_w = drone_w + sat_w + 20

    combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

    drone_y_offset = (combined_h - drone_h) // 2

    combined[drone_y_offset : drone_y_offset + drone_h, :drone_w] = drone_bgr
    combined[0:sat_h, drone_w + 20 : drone_w + 20 + sat_w] = satellite_bgr

    x1, y1, x2, y2 = [int(c) for c in bbox_scaled]
    sat_x1 = drone_w + 20 + x1
    sat_y1 = y1
    sat_x2 = drone_w + 20 + x2
    sat_y2 = y2

    cv2.rectangle(combined, (sat_x1, sat_y1), (sat_x2, sat_y2), (0, 0, 255), 4)

    cv2.circle(combined, (sat_x1, sat_y1), HANDLE_SIZE, (0, 255, 255), -1)
    cv2.circle(combined, (sat_x2, sat_y2), HANDLE_SIZE, (0, 255, 255), -1)
    cv2.circle(combined, (sat_x1, sat_y2), HANDLE_SIZE, (0, 255, 255), -1)
    cv2.circle(combined, (sat_x2, sat_y1), HANDLE_SIZE, (0, 255, 255), -1)

    cv2.putText(
        combined,
        f"Key: {key}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        combined,
        f"Bbox: [{x1}, {y1}, {x2}, {y2}]",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        combined,
        f"Drone: {drone_w}x{drone_h}",
        (10, combined_h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        combined,
        f"Satellite: {sat_w}x{sat_h}",
        (drone_w + 30, combined_h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        combined,
        "Enter=Save&Next, n=Next, p=Prev, r=Reset, q=Quit",
        (10, combined_h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return combined


def main():
    keys, _ = load_paths_from_txt(INPUT_TXT)
    keys = sorted(keys)
    logger.info(f"Found {len(keys)} unique keys in {INPUT_TXT}")

    existing_data = load_existing_json(OUTPUT_JSON)
    remaining_keys = [k for k in keys if k not in existing_data]
    logger.info(f"Remaining keys to process: {len(remaining_keys)}")

    if not remaining_keys:
        logger.info("All images have been processed.")
        return

    results = dict(existing_data)
    current_idx = 0
    current_key = None
    original_drone = None
    original_satellite = None
    scaled_satellite = None
    current_bbox_scaled = None
    original_bbox_scaled = None
    scale_factor = 1.0
    pad_x = 0
    pad_y = 0

    dragging = False
    drag_start = None
    bbox_start = None
    resize_edge = None
    image_loaded = False

    WINDOW_NAME = "BBox Fine-tuning"

    def mouse_callback(event, x, y, flags, param):
        nonlocal \
            dragging, \
            drag_start, \
            bbox_start, \
            current_bbox_scaled, \
            resize_edge, \
            image_loaded

        if not image_loaded or current_bbox_scaled is None or scaled_satellite is None:
            return

        drone_w = original_drone.size[0] if original_drone else 0

        x1, y1, x2, y2 = current_bbox_scaled
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        margin = HANDLE_SIZE + 5

        sat_x1 = x1
        sat_y1 = y1
        sat_x2 = x2
        sat_y2 = y2

        if event == cv2.EVENT_LBUTTONDOWN:
            if (
                sat_x1 - margin <= x <= sat_x1 + margin
                and sat_y1 - margin <= y <= sat_y1 + margin
            ):
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "tl"
            elif (
                sat_x2 - margin <= x <= sat_x2 + margin
                and sat_y2 - margin <= y <= sat_y2 + margin
            ):
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "br"
            elif (
                sat_x2 - margin <= x <= sat_x2 + margin
                and sat_y1 - margin <= y <= sat_y1 + margin
            ):
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "tr"
            elif (
                sat_x1 - margin <= x <= sat_x1 + margin
                and sat_y2 - margin <= y <= sat_y2 + margin
            ):
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "bl"
            elif (
                sat_x1 - margin <= x <= sat_x2 + margin
                and sat_y1 - margin <= y <= sat_y1 + margin
            ):
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "t"
            elif (
                sat_x1 - margin <= x <= sat_x2 + margin
                and sat_y2 - margin <= y <= sat_y2 + margin
            ):
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "b"
            elif (
                sat_x1 <= x <= sat_x1 + margin
                and sat_y1 - margin <= y <= sat_y2 + margin
            ):
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "l"
            elif (
                sat_x2 - margin <= x <= sat_x2 + margin
                and sat_y1 - margin <= y <= sat_y2 + margin
            ):
                dragging = True
                drag_start = (x, y)
                bbox_start = list(current_bbox_scaled)
                resize_edge = "r"
            elif sat_x1 <= x <= sat_x2 and sat_y1 <= y <= sat_y2:
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
                        x2_s = min(scaled_satellite.size[0], max(x2_s + dx, x1_s + 1))
                    if resize_edge in ("br", "bl", "b"):
                        y2_s = min(scaled_satellite.size[1], max(y2_s + dy, y1_s + 1))

                if resize_edge == "tr":
                    x2_s = min(scaled_satellite.size[0], max(x2_s + dx, x1_s + 1))
                    y1_s = max(0, min(y1_s + dy, y2_s - 1))

                if resize_edge == "bl":
                    x1_s = max(0, min(x1_s + dx, x2_s - 1))
                    y2_s = min(scaled_satellite.size[1], max(y2_s + dy, y1_s + 1))

                if resize_edge is None:
                    width = x2_s - x1_s
                    height = y2_s - y1_s
                    x1_s = max(0, min(x1_s + dx, scaled_satellite.size[0] - width))
                    y1_s = max(0, min(y1_s + dy, scaled_satellite.size[1] - height))
                    x2_s = x1_s + width
                    y2_s = y1_s + height

                current_bbox_scaled = [x1_s, y1_s, x2_s, y2_s]
                draw_and_show()

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            drag_start = None
            bbox_start = None
            resize_edge = None

    def draw_and_show():
        if (
            original_drone is None
            or scaled_satellite is None
            or current_bbox_scaled is None
        ):
            return
        combined = draw_combined_image(
            original_drone, scaled_satellite, current_bbox_scaled, current_key
        )
        cv2.imshow(WINDOW_NAME, combined)

    def load_image(idx):
        nonlocal \
            current_idx, \
            current_key, \
            original_drone, \
            original_satellite, \
            scaled_satellite, \
            image_loaded
        nonlocal current_bbox_scaled, original_bbox_scaled, scale_factor, pad_x, pad_y

        current_idx = idx
        current_key = remaining_keys[current_idx]

        drone_path, satellite_path = get_image_paths(current_key)

        if not os.path.exists(drone_path):
            logger.warning(f"Drone image not found: {drone_path}")
            return False
        if not os.path.exists(satellite_path):
            logger.warning(f"Satellite image not found: {satellite_path}")
            return False

        original_drone = Image.open(drone_path).convert("RGB")
        original_satellite = Image.open(satellite_path).convert("RGB")

        scaled_satellite, scale_factor, pad_x, pad_y = apply_letterbox(
            original_satellite, DISPLAY_SIZE
        )

        current_bbox_scaled = convert_bbox_to_scaled(
            INITIAL_BBOX, scale_factor, pad_x, pad_y
        )
        original_bbox_scaled = list(current_bbox_scaled)

        image_loaded = True

        draw_and_show()
        return True

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    logger.info(f"Starting bbox fine-tuning for {len(remaining_keys)} images")
    logger.info("Controls: Enter=Save&Next, n=Next, p=Prev, r=Reset, q=Quit")

    load_image(0)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            save_json(OUTPUT_JSON, results)
            break

        elif key == 13 or key == ord("\n") or key == ord("s"):
            bbox_original = convert_bbox_to_original(
                current_bbox_scaled, scale_factor, pad_x, pad_y
            )
            results[current_key] = bbox_original
            logger.info(f"Saved bbox for {current_key}: {bbox_original}")

            if current_idx < len(remaining_keys) - 1:
                load_image(current_idx + 1)
            else:
                logger.info("Reached last image")
                save_json(OUTPUT_JSON, results)
                break

        elif key == ord("n"):
            if current_idx < len(remaining_keys) - 1:
                load_image(current_idx + 1)

        elif key == ord("p"):
            if current_idx > 0:
                load_image(current_idx - 1)

        elif key == ord("r"):
            current_bbox_scaled = list(original_bbox_scaled)
            draw_and_show()

    cv2.destroyAllWindows()
    logger.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()
