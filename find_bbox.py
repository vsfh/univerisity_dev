import os
import torch
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from loguru import logger
import json
from tqdm import tqdm
try:
    from perceptron.tensorstream import VisionType
    from perceptron.tensorstream.ops import tensor_stream_token_view, modality_mask
    from perceptron.pointing.parser import extract_points
except ImportError:
    logger.error("perceptron package not found. Please ensure it's installed.")
    raise

# ----------------- Load model once -----------------
hf_path = "PerceptronAI/Isaac-0.1"
cache_dir = "/home/SATA4T/gregory/hf_cache"
os.environ['HF_HOME'] = cache_dir
logger.info(f"Loading processor and config from HF checkpoint: {hf_path}")
config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True, use_fast=False, cache_dir=cache_dir)
processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True, cache_dir=cache_dir)
processor.tokenizer = tokenizer

logger.info(f"Loading AutoModelForCausalLM from HF checkpoint: {hf_path}")
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True, cache_dir=cache_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = model.to(device=device, dtype=dtype)
model.eval()

logger.info(f"Model loaded on {device} with dtype {dtype}")

# ----------------- Utility functions -----------------
def document_to_messages(document, vision_token="<image>"):
    messages = []
    images = []
    for item in document:
        itype = item.get("type")
        if itype == "text":
            content = item.get("content")
            if content:
                messages.append({"role": item.get("role", "user"), "content": content})
        elif itype == "image":
            if "content" in item and item["content"] is not None:
                img = PILImage.open(item["content"]).convert("RGB")
                images.append(img)
                messages.append({"role": item.get("role", "user"), "content": vision_token})
    return messages, images

def decode_tensor_stream(tensor_stream, tokenizer):
    token_view = tensor_stream_token_view(tensor_stream)
    mod = modality_mask(tensor_stream)
    text_tokens = token_view[(mod != VisionType.image.value)]
    decoded = tokenizer.decode(text_tokens[0] if len(text_tokens.shape) > 1 else text_tokens)
    return decoded

def visualize_predictions(generated_text, image, output_path="prediction.jpeg"):
    print(generated_text)
    boxes = extract_points(generated_text, expected="box")
    if not boxes:
    #     logger.info("No bounding boxes found in the generated text")
        # image.save(output_path)
        return output_path

    img_width, img_height = image.size
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple"]
    save_boxes = []
    for idx, box in enumerate(boxes):
        color = colors[0]
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

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        save_boxes.append([x1, y1, x2, y2])

        if box.mention:
            text_y = max(y1 - 20, 5)
            text_bbox = draw.textbbox((x1, text_y), box.mention, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, text_y), box.mention, fill="white", font=font)

    img_with_boxes.save(output_path, "JPEG")
    print(save_boxes)
    return output_path, save_boxes

def generate_response(image_path, prompt=""):
    document = [
        {"type": "text", "content": "<hint>BOX</hint>", "role": "user"},
        {"type": "image", "content": image_path, "role": "user"},
        {"type": "text", "content": prompt, "role": "user"},
    ]

    messages, images = document_to_messages(document, vision_token=config.vision_token)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(text)
    inputs = processor(text=text, images=images, return_tensors="pt")
    tensor_stream = inputs["tensor_stream"].to(device)
    input_ids = inputs["input_ids"].to(device)

    _ = decode_tensor_stream(tensor_stream, processor.tokenizer)

    with torch.no_grad():
        generated_ids = model.generate(
            tensor_stream=tensor_stream,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        if images:
            out_name = image_path.replace('image_1024', 'bbox_out')
            vis_path, save_boxes = visualize_predictions(generated_text, images[0], output_path=out_name)
            return generated_text, vis_path, save_boxes
        else:
            return generated_text, None

# ----------------- Main loop over folder -----------------
if __name__ == "__main__":
    input_folder = "/home/SATA4T/gregory/data/image_1024/"  # folder of images
    output_text_file = "results.txt"
    # prompt = """Your task is to act as an expert in aerial image analysis. Given an aerial image, identify every **significant, distinct, and non-overlapping regions**.

    #         A significant region is a key landmark, a main building, or a distinct functional area. Your goal is to provide a balanced overview of the image's most important components.

    #         ### **Guidelines for Selecting Regions:**

    #         1.  **Significance:** Prioritize regions that are architecturally or functionally important. Good examples include:
    #             * **Primary Buildings:** The main academic hall, a stadium, an industrial complex, a church, a uniquely shaped building (e.g., circular or domed), or a large residential complex.
    #             * **Distinct Zones:** A cohesive sports complex (e.g., a football field and its track), a central campus quad or courtyard, a large and clearly defined parking facility, or a main plaza.
    #             * **Key Green Spaces:** A central park, a large manicured lawn, or a botanical garden, not just scattered patches of grass.

    #         2.  **Size and Granularity:**
    #             * **Avoid Trivial Objects:** Do not select small, individual items. **Ignore** single cars, small sheds, individual trees, or minor structures. Focus on entire buildings or functionally distinct areas.
    #             * **Avoid Overly Broad Selections:** Do not select the entire image or vast, undifferentiated areas (like a massive, empty field or an entire suburban neighborhood) as a single region. Break down larger complexes into their main constituent parts. For example, within a university campus, identify the main library, the stadium, and the student center as separate regions rather than the entire campus as one.

    #         3.  **Non-Overlapping Rule:**
    #             * The 8 bounding boxes you provide **must not overlap** with each other. Each region should be entirely separate.

    #         4. **Grouping Rule:** 
    #             *If there are close building groups or regions with similar style, label them with a single bounding box


    #         ### **Output Format:**

    #         Provide a list containing exactly 8 bounding box arrays in the format `[x_min, y_min, x_max, y_max]`.
    #         """
    # prompt = "give me the top 8 region people will notice on the map at the first look. think step by step"
    prompt = 'large industrial complex with a prominent beige-roofed building'
    out_boxes_dict = {}
    for file in tqdm(os.listdir(input_folder)):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            img_path = os.path.join(input_folder, file)
            img_path = '/home/SATA4T/gregory/data/image_1024/0001.png'
            logger.info(f"Processing {img_path}")
            text, vis_path, save_boxes = generate_response(img_path, prompt=prompt)
            out_boxes_dict[file] = save_boxes
            logger.info(f"Saved visualization to {vis_path}")
        break
    logger.info("Processing complete.")
    with open("boxes_output_new.json", "w") as jf:
        json.dump(out_boxes_dict, jf)
    