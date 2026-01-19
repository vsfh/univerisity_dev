import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import re
import shutil

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
CACHE_DIR = "/home/SATA4T/gregory/hf_cache"
SATELLITE_DIR = "/data/feihong/image_1024"
TRAIN_FILE = "/data/feihong/ckpt/train.txt"
TEST_FILE = "/data/feihong/ckpt/test.txt"
TEXT_FILE = "/data/feihong/drone_text_single_long.json"
BBOX_FILE = "/data/feihong/univerisity_dev/bbox_results.json"

NUM_EPOCHS = 30
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "runs/train_qwen"
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIS_DIR = os.path.join(OUTPUT_DIR, "vis")
os.makedirs(VIS_DIR, exist_ok=True)
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR


def extract_id_from_path(path):
    parts = path.split("/")
    for part in reversed(parts):
        if part.isdigit():
            return part
    return None


def load_bbox_annotations(bbox_file):
    if os.path.exists(bbox_file):
        with open(bbox_file, "r") as f:
            return json.load(f)
    return {}


def load_text_annotations(text_file):
    if os.path.exists(text_file):
        with open(text_file, "r") as f:
            return json.load(f)
    return {}


class CrossViewDataset(Dataset):
    def __init__(
        self,
        split_file,
        processor,
        bbox_annotations,
        text_annotations,
        use_text=True,
        mode="train",
    ):
        self.image_paths = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.image_paths.append(line)

        self.processor = processor
        self.bbox_annotations = bbox_annotations
        self.text_annotations = text_annotations
        self.use_text = use_text
        self.mode = mode

        self.satellite_size = (1024, 1024)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        drone_path = self.image_paths[idx]

        drone_id = extract_id_from_path(drone_path)
        if drone_id is None:
            drone_id = os.path.basename(drone_path).split(".")[0].replace("image-", "")

        satellite_path = os.path.join(SATELLITE_DIR, f"{drone_id}.png")

        try:
            drone_img = Image.open(drone_path).convert("RGB")
            satellite_img = Image.open(satellite_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Warning: Could not load image pair for {drone_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        bbox = self.bbox_annotations.get(f"{drone_id}.png", None)
        if bbox is None:
            bbox = self.bbox_annotations.get(f"{drone_id}.jpeg", None)

        text_key = f"{drone_id}_01"
        text_desc = ""
        if self.use_text and text_key in self.text_annotations:
            text_desc = self.text_annotations[text_key]

        return {
            "drone_image": drone_img,
            "satellite_image": satellite_img,
            "text": text_desc,
            "bbox": bbox,
            "id": f"{drone_id}.png",
            "drone_path": drone_path,
        }


def collate_fn(batch):
    drone_images = []
    satellite_images = []
    texts = []
    bboxes = []
    ids = []
    drone_paths = []

    for item in batch:
        drone_images.append(item["drone_image"])
        satellite_images.append(item["satellite_image"])
        texts.append(item["text"])
        bboxes.append(item["bbox"])
        ids.append(item["id"])
        drone_paths.append(item["drone_path"])

    return {
        "drone_images": drone_images,
        "satellite_images": satellite_images,
        "texts": texts,
        "bboxes": bboxes,
        "ids": ids,
        "drone_paths": drone_paths,
    }


def format_bbox_for_output(bbox, image_size=(1024, 1024)):
    if bbox is None:
        return "[0, 0, 0, 0]"
    x1, y1, x2, y2 = bbox
    width, height = image_size
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    return f"[{x1}, {y1}, {x2}, {y2}]"


def train_one_epoch(model, dataloader, optimizer, processor, epoch, writer, device):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        drone_imgs = batch["drone_images"]
        sat_imgs = batch["satellite_images"]
        texts = batch["texts"]
        bboxes = batch["bboxes"]

        batch_strs = []
        target_strs = []

        for i in range(len(drone_imgs)):
            drone_pil = drone_imgs[i]
            sat_pil = sat_imgs[i]
            text_desc = texts[i] if texts[i] else "No description available."
            target_bbox = format_bbox_for_output(bboxes[i])

            user_msg = f"<|image_1|>\n<|image_2|>\nDescription: {text_desc}\n\nLocate this building in the satellite image. Provide the bounding box in format: [x1, y1, x2, y2] for a 1024x1024 image."
            assistant_msg = target_bbox

            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]

            text_str = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            batch_strs.append(text_str)
            target_strs.append(assistant_msg)

        try:
            inputs = processor(
                text=batch_strs,
                images=[drone_imgs, sat_imgs],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
        except Exception as e:
            print(f"Error in processor call: {e}")
            continue

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        if pixel_values is not None and isinstance(pixel_values, list):
            pixel_values = [pv.to(device) for pv in pixel_values]
        elif pixel_values is not None:
            pixel_values = pixel_values.to(device)

        labels = input_ids.clone()

        for i, target in enumerate(target_strs):
            try:
                target_tokens = processor.tokenizer.encode(
                    target, add_special_tokens=False
                )
                target_start = len(
                    processor.tokenizer.encode(
                        processor.apply_chat_template(
                            [{"role": "user", "content": user_msg}],
                            tokenize=False,
                            add_generation_prompt=False,
                        ),
                        add_special_tokens=False,
                    )
                )
                for j, tok_id in enumerate(target_tokens):
                    if target_start + j < labels.shape[1]:
                        labels[i, target_start + j] = tok_id
            except:
                pass

        labels = labels.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )

        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if batch_idx % 100 == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Train/loss", loss.item(), step)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate(model, dataloader, processor, epoch, writer, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    num_samples = 0

    all_preds = []
    all_gts = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            drone_imgs = batch["drone_images"]
            sat_imgs = batch["satellite_images"]
            texts = batch["texts"]
            bboxes = batch["bboxes"]
            ids = batch["ids"]

            batch_strs = []

            for i in range(len(drone_imgs)):
                text_desc = texts[i] if texts[i] else "No description available."

                user_msg = f"<|image_1|>\n<|image_2|>\nDescription: {text_desc}\n\nLocate this building in the satellite image. Provide the bounding box in format: [x1, y1, x2, y2] for a 1024x1024 image."

                messages = [{"role": "user", "content": user_msg}]
                text_str = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_strs.append(text_str)

            try:
                inputs = processor(
                    text=batch_strs,
                    images=[drone_imgs, sat_imgs],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                )
            except Exception as e:
                print(f"Error in evaluation processor call: {e}")
                continue

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            pixel_values = inputs.get("pixel_values", None)
            image_grid_thw = inputs.get("image_grid_thw", None)

            if pixel_values is not None and isinstance(pixel_values, list):
                pixel_values = [pv.to(device) for pv in pixel_values]
            elif pixel_values is not None:
                pixel_values = pixel_values.to(device)

            try:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

                generated_texts = processor.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                for i, gen_text in enumerate(generated_texts):
                    gt_bbox = bboxes[i]
                    pred_bbox = parse_bbox_from_response(gen_text)

                    if pred_bbox is not None and gt_bbox is not None:
                        iou = compute_iou(pred_bbox, gt_bbox)
                        total_iou += iou

                        all_preds.append({"id": ids[i], "bbox": pred_bbox})
                        all_gts.append({"id": ids[i], "bbox": gt_bbox})
                        all_ids.append(ids[i])

                        if epoch % 5 == 0 and len(all_preds) < 20:
                            visualize_prediction(
                                sat_imgs[i],
                                pred_bbox,
                                gt_bbox,
                                os.path.join(
                                    VIS_DIR,
                                    f"{ids[i].replace('.png', '')}_epoch{epoch}.png",
                                ),
                            )

                    num_samples += 1

            except Exception as e:
                print(f"Error in generation: {e}")
                continue

    avg_iou = total_iou / max(num_samples, 1)

    if writer is not None:
        step = (epoch + 1) * len(dataloader)
        writer.add_scalar("Eval/IoU", avg_iou, step)

    results = {
        "predictions": all_preds,
        "ground_truths": all_gts,
        "average_iou": avg_iou,
    }

    print(f"Evaluation - Average IoU: {avg_iou:.4f} ({num_samples} samples)")
    return results, avg_iou


def parse_bbox_from_response(text):
    try:
        text = text.strip()
        text = re.sub(r".*?\[", "[", text)
        nums = re.findall(r"\d+", text)
        if len(nums) >= 4:
            return [int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])]
    except:
        pass
    return None


def compute_iou(pred_bbox, gt_bbox):
    x1_p, y1_p, x2_p, y2_p = pred_bbox
    x1_g, y1_g, x2_g, y2_g = gt_bbox

    inter_x1 = max(x1_p, x1_g)
    inter_y1 = max(y1_p, y1_g)
    inter_x2 = min(x2_p, x2_g)
    inter_y2 = min(y2_p, y2_g)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    pred_area = (x2_p - x1_p) * (y2_p - y1_p)
    gt_area = (x2_g - x1_g) * (y2_g - y1_g)

    union_area = pred_area + gt_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def visualize_prediction(satellite_img, pred_bbox, gt_bbox, output_path):
    img = satellite_img.copy()
    draw = ImageDraw.Draw(img)

    if gt_bbox:
        x1_g, y1_g, x2_g, y2_g = gt_bbox
        draw.rectangle([x1_g, y1_g, x2_g, y2_g], outline="green", width=3)

    if pred_bbox:
        x1_p, y1_p, x2_p, y2_p = pred_bbox
        draw.rectangle([x1_p, y1_p, x2_p, y2_p], outline="red", width=3)

    img.save(output_path, "PNG")


def save_model(model, processor, output_dir, epoch):
    save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
    os.makedirs(save_path, exist_ok=True)

    try:
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"Saved checkpoint to {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto", cache_dir=CACHE_DIR
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    print(f"Model loaded on device(s)")

    bbox_annotations = load_bbox_annotations(BBOX_FILE)
    print(f"Loaded {len(bbox_annotations)} bbox annotations")

    text_annotations = load_text_annotations(TEXT_FILE)
    print(f"Loaded {len(text_annotations)} text annotations")

    train_dataset = CrossViewDataset(
        TRAIN_FILE,
        processor,
        bbox_annotations,
        text_annotations,
        use_text=True,
        mode="train",
    )
    test_dataset = CrossViewDataset(
        TEST_FILE,
        processor,
        bbox_annotations,
        text_annotations,
        use_text=True,
        mode="test",
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir=LOG_DIR)

    best_iou = 0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, processor, epoch, writer, DEVICE
        )
        writer.add_scalar("Train/epoch_loss", train_loss, epoch)
        print(f"Training loss: {train_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            eval_results, eval_iou = evaluate(
                model, test_loader, processor, epoch, writer, DEVICE
            )

            if eval_iou > best_iou:
                best_iou = eval_iou
                best_epoch = epoch
                save_model(model, processor, OUTPUT_DIR, "best")
                print(f"New best model saved with IoU: {best_iou:.4f}")

            save_model(model, processor, OUTPUT_DIR, epoch + 1)

    save_model(model, processor, OUTPUT_DIR, "final")

    writer.close()

    print(f"\nTraining complete!")
    print(f"Best IoU: {best_iou:.4f} at epoch {best_epoch + 1}")
    print(f"Checkpoints saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
