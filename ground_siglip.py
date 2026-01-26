import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
from tqdm import tqdm
import os
from glob import glob
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import json
from typing import Tuple, List

# --- Configuration ---
MODEL_NAME = "google/siglip-base-patch16-224"
CACHE_DIR = "/data/feihong/hf_cache"
DRONE_VIEW_FOLDER = "/data/feihong/drone_view"
IMAGE_FOLDER = "/data/feihong/image_1024"
BBOX_FILE = "/data/feihong/univerisity_dev/runs/bbox_isaac.json"
TEXT_FILE = "/data/feihong/drone_text_single_long.json"
LOG_PATH = "../ckpt/ground_siglip"

SAT_ORIG_SIZE = (3840, 2160)
CROP_SIZE = (640, 640)
DRONE_SIZE = (512, 512)

NUM_EPOCHS = 40
BATCH_SIZE = 12
LEARNING_RATE = 1e-5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECTION_DIM = 768


def format_satellite_img_bbox(
    image: Image.Image,
    bbox: List[int],
    mode: str = "train",
    target_size: Tuple[int, int] = CROP_SIZE,
) -> Tuple[Image.Image, List[float]]:

    x1, y1, x2, y2 = bbox
    width, height = image.size
    crop_size = height // 5 * 4  # 1296

    min_left = (width / 2) - crop_size
    max_left = width / 2
    min_top = (height / 2) - crop_size
    max_top = height / 2

    min_left = max(0, min_left)
    min_top = max(0, min_top)
    max_left = min(width - crop_size, max_left)
    max_top = min(height - crop_size, max_top)

    if max_left < min_left:
        max_left = min_left
    if max_top < min_top:
        max_top = min_top

    min_left = max(min_left, x2 - crop_size)
    min_top = max(min_top, y2 - crop_size)
    max_left = min(max_left, x1)
    max_top = min(max_top, y1)

    left = random.uniform(min_left, max_left)
    top = random.uniform(min_top, max_top)

    right = left + crop_size
    bottom = top + crop_size

    image = image.crop((left, top, right, bottom))
    image = image.resize(target_size)

    new_x1 = (x1 - left) / crop_size
    new_y1 = (y1 - top) / crop_size
    new_x2 = (x2 - left) / crop_size
    new_y2 = (y2 - top) / crop_size
    return image, [new_x1, new_y1, new_x2, new_y2]


def resize_drone_image(
    image: Image.Image, target_size: Tuple[int, int] = (512, 512)
) -> Image.Image:
    """
    Resize drone image to target size.

    Args:
        image: Input drone image
        target_size: Target (width, height)

    Returns:
        Resized image
    """
    return image.resize(target_size, Image.Resampling.BILINEAR)


def compute_iou(pred_bbox: List[float], gt_bbox: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between predicted and ground truth bboxes.

    Args:
        pred_bbox: Predicted bounding box [x1, y1, x2, y2] normalized [0, 1]
        gt_bbox: Ground truth bounding box [x1, y1, x2, y2] normalized [0, 1]

    Returns:
        IoU score between 0 and 1
    """
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


class GroundSiglipDataset(Dataset):
    def __init__(
        self,
        image_pairs,
        processor,
        processor_sat,
        tokenizer,
        img_to_text_dict,
        bbox_dict,
        mode="train",
    ):
        self.image_paths = image_pairs
        self.processor = processor
        self.processor_sat = processor_sat
        self.tokenizer = tokenizer
        self.img_to_text_dict = img_to_text_dict
        self.bbox_dict = bbox_dict
        self.mode = mode
        self.remove = [
            "**",
            "\n",
            "noun",
            "phrases",
            "Phrase",
            "Noun",
            "Summary",
            "Environment",
            "32 tokens",
            "Description",
            "()",
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        query_path, search_path = self.image_paths[idx]

        if self.mode == "train":
            choice = []
            for number in ["01", "21", "31", "41", "51"]:
                new_query_path = query_path.replace("01", number)
                if not os.path.exists(new_query_path):
                    continue
                choice.append(new_query_path)

            if choice:
                query_path = random.sample(choice, 1)[0]

        name = (
            query_path.split("/")[-2]
            + "_"
            + query_path.split("/")[-1].split(".")[0][-2:]
        )
        text_description = self.img_to_text_dict.get(name, "")

        for noun in self.remove:
            text_description = text_description.replace(noun, "")

        try:
            query_image = Image.open(query_path).convert("RGB")
            search_image = Image.open(search_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping this item.")
            return self.__getitem__((idx + 1) % len(self))

        query_image = resize_drone_image(query_image, DRONE_SIZE)

        base_name = name
        bbox_key = f"{base_name}.png"
        bbox = self.bbox_dict.get(bbox_key, None)
        if bbox is None:
            bbox = self.bbox_dict.get(bbox_key.replace(".png", ".jpeg"), None)

        bbox = (1536, 656, 2268, 1374)

        search_image, normalized_bbox = format_satellite_img_bbox(
            search_image, bbox, mode=self.mode, target_size=CROP_SIZE
        )

        text_inputs = self.tokenizer(
            [text_description],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        query_inputs = self.processor(images=query_image, return_tensors="pt")
        search_inputs = self.processor_sat(images=search_image, return_tensors="pt")

        return {
            "drone_pixel_values": query_inputs["pixel_values"][0],
            "sat_pixel_values": search_inputs["pixel_values"][0],
            "input_ids": text_inputs["input_ids"][0],
            "bbox": torch.tensor(normalized_bbox, dtype=torch.float32),
            "name": name,
        }


class GroundSiglipEncoder(nn.Module):
    def __init__(self, model_name, proj_dim=768):
        super().__init__()

        try:
            self.model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.vision_model = self.model.vision_model
            self.text_model = self.model.text_model

            self.feature_dim = self.vision_model.config.hidden_size
            self.text_feature_dim = self.text_model.config.hidden_size
            self.num_patches = (640 // 16) ** 2  # 1600 patches for 640x640 image
        except Exception as e:
            print(f"Error loading SIGLIP model: {e}")
            raise

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.text_feature_dim * 2, proj_dim),
        )

        self.pool = nn.AdaptiveAvgPool2d((20, 20))
        self.position_embedding = nn.Embedding(400, PROJECTION_DIM)
        self.register_buffer("position_ids", torch.arange(400).expand((1, -1)), persistent=False)

        self.drone_projector = nn.Linear(self.feature_dim, proj_dim)

        self.fusion_fc = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.ReLU(),
        )

        self.bbox_head = nn.Sequential(
            nn.Conv1d(proj_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 4, kernel_size=1),
        )

    def sat_forward(self, pixel_values):
        """
        Extract satellite image features from last hidden state without pooling.

        Args:
            pixel_values: Satellite image tensor (B, 3, H, W)

        Returns:
            sat_feat: (B, N, D) where N = num_patches, D = hidden_size
        """
        embedding = self.vision_model.embeddings.patch_embedding(pixel_values)
        embedding = self.pool(embedding).flatten(2).transpose(1, 2) + self.position_embedding(self.position_ids)
        sat_feat = self.vision_model.encoder(embedding).last_hidden_state
        return sat_feat

    def query_forward(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values)
        pooler_output = vision_outputs.pooler_output
        projected = self.drone_projector(pooler_output)
        return projected

    def text_forward(self, input_ids, attention_mask=None):
        text_outputs = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooler_output = text_outputs.pooler_output
        proj_feature = self.text_projector(pooler_output)
        return proj_feature

    def forward(
        self, drone_pixel_values, sat_pixel_values, input_ids, attention_mask=None
    ):
        drone_feat = self.query_forward(drone_pixel_values)
        sat_feat = self.sat_forward(sat_pixel_values)
        text_feat = self.text_forward(input_ids, attention_mask)

        B, N, D = sat_feat.shape

        drone_expanded = drone_feat.unsqueeze(1).expand(-1, N, -1)
        text_expanded = text_feat.unsqueeze(1).expand(-1, N, -1)

        fused = torch.cat([sat_feat, drone_expanded, text_expanded], dim=-1)
        fused = self.fusion_fc(fused)

        fused = fused.permute(0, 2, 1)
        bbox_pred = self.bbox_head(fused)
        bbox_pred = bbox_pred.permute(0, 2, 1)

        bbox_pred = bbox_pred.mean(dim=1)
        bbox_pred = torch.sigmoid(bbox_pred)

        return bbox_pred, drone_feat, text_feat


def compute_bbox_loss(
    pred_bbox: torch.Tensor, target_bbox: torch.Tensor
) -> torch.Tensor:
    """
    Compute bbox loss: L1 loss + IoU loss.

    Args:
        pred_bbox: Predicted bounding box [B, 4] normalized [0, 1]
        target_bbox: Target bounding box [B, 4] normalized [0, 1]

    Returns:
        Combined loss value
    """
    l1_loss = F.l1_loss(pred_bbox, target_bbox)

    pred_x1 = pred_bbox[:, 0]
    pred_y1 = pred_bbox[:, 1]
    pred_x2 = pred_bbox[:, 2]
    pred_y2 = pred_bbox[:, 3]

    target_x1 = target_bbox[:, 0]
    target_y1 = target_bbox[:, 1]
    target_x2 = target_bbox[:, 2]
    target_y2 = target_bbox[:, 3]

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    union_area = pred_area + target_area - inter_area + 1e-6

    iou = inter_area / union_area
    iou_loss = -torch.log(iou + 1e-6)

    total_loss = 0.5 * l1_loss + 0.5 * iou_loss.mean()

    return total_loss


def main(save_path):
    exp_name = save_path.split("/")[-1] if save_path else "default_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Gathering image pairs...")
    image_pairs = []
    for query_path in tqdm(glob(f"{DRONE_VIEW_FOLDER}/*/*/image-01.jpeg")):
        name = query_path.split("/")[-2]
        search_path = f"{IMAGE_FOLDER}/{name}.png"
        if os.path.exists(search_path):
            image_pairs.append((query_path, search_path))
    print(f"Found {len(image_pairs)} valid pairs.")

    print("Loading models and processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor_sat = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640})

    model = GroundSiglipEncoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(
        DEVICE
    )
    model.load_state_dict(torch.load("../ckpt/train_siglip_proj/best.pth", map_location="cpu"), strict=False)

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    img_to_text_dict = json.load(open(TEXT_FILE, "r"))
    bbox_dict = json.load(open(BBOX_FILE, "r"))

    print("Setting up dataset and dataloader...")
    train_image_pairs = []
    with open("/data/feihong/ckpt/train.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            if os.path.exists(search_path):
                train_image_pairs.append((query_path, search_path))

    test_image_pairs = []
    with open("/data/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            if os.path.exists(search_path):
                test_image_pairs.append((query_path, search_path))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = GroundSiglipDataset(
        image_pairs=train_image_pairs,
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        img_to_text_dict=img_to_text_dict,
        bbox_dict=bbox_dict,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4
    )

    test_dataset = GroundSiglipDataset(
        image_pairs=test_image_pairs,
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        img_to_text_dict=img_to_text_dict,
        bbox_dict=bbox_dict,
        mode="test",
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4
    )

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")
    best_iou = 0
    not_update = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_bbox_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for i, batch in enumerate(progress_bar):
            drone_pixel_values = batch["drone_pixel_values"].to(DEVICE)
            sat_pixel_values = batch["sat_pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            target_bbox = batch["bbox"].to(DEVICE)

            bbox_pred, drone_feat, text_feat = model(
                drone_pixel_values, sat_pixel_values, input_ids
            )

            bbox_loss = compute_bbox_loss(bbox_pred, target_bbox)

            loss = bbox_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "bbox": f"{bbox_loss.item():.4f}",
                }
            )
            writer.add_scalar(
                "Loss/train_batch", loss.item(), epoch * len(train_dataloader) + i
            )

        avg_loss = total_loss / len(train_dataloader)
        avg_bbox_loss = total_bbox_loss / len(train_dataloader)

        model.eval()
        val_metrics = validation(model, test_dataloader, epoch)
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Loss/val_bbox_epoch", val_metrics["bbox_loss"], epoch)
        writer.add_scalar("Metrics/mean_iou", val_metrics["mean_iou"], epoch)
        writer.add_scalar("Metrics/iou@0.5", val_metrics["iou_at_0.5"], epoch)

        print(
            f"Epoch {epoch + 1} finished. "
            f"Avg train loss: {avg_loss:.4f} (bbox: {avg_bbox_loss:.4f}). "
            f"Val mean IoU: {val_metrics['mean_iou']:.4f}, IoU@0.5: {val_metrics['iou_at_0.5']:.4f}"
        )

        if val_metrics["mean_iou"] > best_iou:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/best.pth")
            best_iou = val_metrics["mean_iou"]
            not_update = 0
        else:
            not_update += 1
        if not_update > 5:
            print("Validation loss not improving. Stopping early.")
            break
    print(f"Training complete. Best IoU: {best_iou:.4f}")
    writer.close()


def validation(model, loader, epoch):
    progress_bar = tqdm(loader, desc=f"Valid {epoch + 1}/{NUM_EPOCHS}")
    total_bbox_loss = 0
    all_ious = []

    for batch in progress_bar:
        drone_pixel_values = batch["drone_pixel_values"].to(DEVICE)
        sat_pixel_values = batch["sat_pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        target_bbox = batch["bbox"].to(DEVICE)

        with torch.no_grad():
            bbox_pred, drone_feat, text_feat = model(
                drone_pixel_values, sat_pixel_values, input_ids
            )

            bbox_loss = compute_bbox_loss(bbox_pred, target_bbox)

            total_bbox_loss += bbox_loss.item()

            for j in range(bbox_pred.shape[0]):
                pred = bbox_pred[j].cpu().tolist()
                gt = target_bbox[j].cpu().tolist()
                iou = compute_iou(pred, gt)
                all_ious.append(iou)

    avg_bbox_loss = total_bbox_loss / len(loader)

    mean_iou = np.mean(all_ious) if all_ious else 0.0
    iou_at = (
        sum(1 for iou in all_ious if iou >= 0.5) / len(all_ious) if all_ious else 0.0
    )

    return {
        "bbox_loss": avg_bbox_loss,
        "mean_iou": mean_iou,
        "iou_at_0.5": iou_at,
    }


if __name__ == "__main__":
    save_dir = LOG_PATH

    if os.path.exists(save_dir):
        print(f"Experiment directory '{save_dir}' already exists.")
        print("Re-using directory. Logs and checkpoints might be overwritten.")
    else:
        os.makedirs(save_dir)
        print(f"Created experiment directory: {save_dir}")

    main(save_dir)
