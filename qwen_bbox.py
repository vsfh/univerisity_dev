
import json
import torch
import cv2
import os
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# --- SYSTEM CONFIGURATION ---
DEVICE = "cuda:2"  # Target GPU (e.g., "cuda:1", "cuda:2")
CACHE_DIR = "/data/feihong/hf_cache"  # Set your custom storage path
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


# Force HF to use the specific cache directory
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
# ----------------------------

class SatelliteDetector:
    def __init__(self, model_id=MODEL_ID, device=DEVICE):
        self.device = device

        # 1. Memory Optimization: 4-bit quantization for 23GB GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        # 2. Load model from cache with specific device pinning
        print(f"Loading {model_id} onto {device}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": self.device},
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=CACHE_DIR)

    def inference(self, image_path, description):
        # 3. Handle 4K Image Logic (3840x2160)
        # We limit max_pixels for VRAM safety; Qwen3 uses a ratio-based system (0-1000)
        # which ensures the resulting bbox maps accurately back to the 4K original.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path,
                        "max_pixels": 1280 * 1280  # Keeps token count manageable on 23GB
                    },
                    {"type": "text", "text": f"Locate: {description}. Return only JSON: {{\"bbox_2d\": [ymin, xmin, ymax, xmax]}}"}
                ],
            }
        ]

        # Prepare inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # 4. Pure Inference (No extra words)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        return response

    def visualize(self, image_path, response, output_path="result.jpg"):
        try:
            data = json.loads(response)
            bboxes = data.get("bbox_2d", [])
            if not bboxes: return

            img = cv2.imread(image_path) # Read original 4K resolution
            h, w = img.shape[:2]

            if isinstance(bboxes[0], (int, float)): bboxes = [bboxes]

            for box in bboxes:
                ymin, xmin, ymax, xmax = box
                # Map 0-1000 ratio back to 3840x2160 pixels
                l, t = int(xmin * w / 1000), int(ymin * h / 1000)
                r, b = int(xmax * w / 1000), int(ymax * h / 1000)

                cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 5)

            cv2.imwrite(output_path, img)
        except:
            pass

# --- LOOP USAGE EXAMPLE ---
if __name__ == "__main__":

    # Example tasks list

    json_data = json.load(open('/data/feihong/drone_text_long.json', 'r'))
    tasks = [ {'img': f'/data/feihong/image_1024/{k}.png', 'desc': v[24:]} for k,v in json_data.items() ]
    detector = SatelliteDetector()
    res = {}
    num = 0


    for img, desc in json_data.items():
        # 1. Direct Coordinate Output (Only raw JSON string)
        coords_json = detector.inference(f'/data/feihong/image_1024/{img}.png', desc[24:])
        print(coords_json)
        res[img] = coords_json
        num += 1
        if num > 50:
            break
        # 2. Optional Visualization
        # detector.visualize(task["img"], coords_json, f"out_{task['img']}")
    json.dump(res, open('res.json', 'w'))