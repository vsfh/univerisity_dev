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

# --- Configuration ---
MODEL_NAME = "google/siglip-base-patch16-224"
CACHE_DIR = "/data/feihong/hf_cache"
DRONE_VIEW_FOLDER = "/data/feihong/drone_view"
IMAGE_FOLDER = "/data/feihong/image_1024"
BBOX_FILE = "/data/feihong/univerisity_dev/runs/bbox_isaac.json"
TEXT_FILE = "/data/feihong/drone_text_single_long.json"

SAT_ORIG_SIZE = (3840, 2160)
CROP_SIZE = (640, 640)
DRONE_SIZE = (512, 512)

NUM_EPOCHS = 40
BATCH_SIZE = 20
LEARNING_RATE = 1e-5
BBOX_LOSS_WEIGHT = 0.1
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECTION_DIM = 768


def format_satellite_img_bbox(image, bbox, mode="train", target_size=CROP_SIZE):
    x1, y1, x2, y2 = bbox
    width, height = image.size
    crop_size = height // 5 * 4

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


def resize_drone_image(image, target_size=DRONE_SIZE):
    return image.resize(target_size, Image.Resampling.BILINEAR)


class TargetSearchDataset(Dataset):
    def __init__(
        self,
        image_pairs,
        processor,
        processor_sat,
        tokenizer,
        img_to_text_dict,
        bbox_dict=None,
        mode="train",
    ):
        self.image_paths = image_pairs
        self.processor = processor
        self.processor_sat = processor_sat
        self.tokenizer = tokenizer
        self.img_to_text_dict = img_to_text_dict
        self.bbox_dict = bbox_dict

        self.crop_size = 2160 // 5 * 3  # 1296
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
        bbox = None
        if self.bbox_dict:
            bbox = self.bbox_dict.get(bbox_key, None)
            if bbox is None:
                bbox = self.bbox_dict.get(bbox_key.replace(".png", ".jpeg"), None)
        if bbox is None:
            bbox = (1536, 656, 2268, 1374)

        if self.mode == "train":
            augmented_crop_image, normalized_bbox = format_satellite_img_bbox(
                search_image, bbox, mode=self.mode, target_size=CROP_SIZE
            )
            center_x = (normalized_bbox[0] + normalized_bbox[2]) / 2
            center_y = (normalized_bbox[1] + normalized_bbox[3]) / 2

            col_idx = min(int(center_x * 3), 2)
            row_idx = min(int(center_y * 3), 2)
            index = row_idx * 3 + col_idx
        else:
            augmented_crop_image = search_image.crop((840, 0, 3000, 2160)).resize(
                (640, 640)
            )
            normalized_bbox = [0.5, 0.5, 0.5, 0.5]
            index = 4

        text_inputs = self.tokenizer(
            [text_description],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        query_inputs = self.processor(images=query_image, return_tensors="pt")
        search_inputs = self.processor_sat(
            images=augmented_crop_image, return_tensors="pt"
        )

        return {
            "target_pixel_values": query_inputs["pixel_values"][0],
            "search_pixel_values": search_inputs["pixel_values"][0],
            "input_ids": text_inputs["input_ids"][0],
            "index": index,
            "bbox": torch.tensor(normalized_bbox, dtype=torch.float32),
        }


class Encoder(nn.Module):
    def __init__(self, model_name, proj_dim=768):
        super().__init__()

        try:
            self.model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.vision_model = self.model.vision_model
            self.text_model = self.model.text_model

            self.feature_dim = self.vision_model.config.hidden_size
            self.text_feature_dim = self.text_model.config.hidden_size
        except Exception as e:
            print(f"Error loading SIGLIP model: {e}")
            raise

        pool_size = 20
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.position_embedding = nn.Embedding(9, PROJECTION_DIM)
        self.register_buffer(
            "position_ids", torch.arange(9).expand((1, -1)), persistent=False
        )

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.text_feature_dim * 2, proj_dim),
        )

        self.bbox_mlp = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
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
        embedding = self.pool(embedding).flatten(2).transpose(
            1, 2
        ) + self.position_embedding(self.position_ids)
        sat_feat = self.vision_model.encoder(embedding).last_hidden_state
        return sat_feat

    def query_forward(self, pixel_values):
        pooled_features = self.vision_model(pixel_values).pooler_output
        return pooled_features

    def text_forward(self, input_ids, attention_mask=None):
        text_outputs = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooler_output = text_outputs.pooler_output
        proj_feature = self.text_projector(pooler_output)
        return proj_feature

    def bbox_forward(self, anchor_pixel_values, search_pixel_values):
        """
        Combined forward for bbox prediction + retrieval.

        Args:
            anchor_pixel_values: Query image tensor (B, 3, H, W)
            search_pixel_values: Search image tensor (B, 3, H, W)

        Returns:
            bbox_pred: B×4 (normalized bbox coordinates [x1, y1, x2, y2])
            anchor_feats: B×768 (query features for InfoNCE)
            grid_feats: B×N×768 (pooled grid features for InfoNCE)
        """
        B = anchor_pixel_values.shape[0]

        anchor_feats = self.query_forward(anchor_pixel_values)

        grid_features = self.sat_forward(search_pixel_values)

        anchor_norm = F.normalize(anchor_feats, p=2, dim=1)
        grid_norm = F.normalize(grid_features, p=2, dim=2)

        similarity = torch.bmm(grid_norm, anchor_norm[..., None])

        combined = torch.cat(
            [
                grid_features,
                similarity * grid_features,
            ],
            dim=-1,
        )

        combined_pooled = combined.mean(dim=1)
        bbox_pred = self.bbox_mlp(combined_pooled).sigmoid()

        # pooled = self._pool_grid_features(grid_features)
        return bbox_pred, anchor_feats, grid_features

    def _pool_grid_features(self, vision_features):
        B, N, D = vision_features.shape
        H = W = int(N**0.5)

        patch_tokens = vision_features.permute(0, 2, 1).reshape(B, D, H, W)
        pooled = self.pool(patch_tokens)
        pooled_features = pooled.flatten(2).permute(0, 2, 1)

        return pooled_features


def info_nce_loss(query_feats, candidate_feats, positive_indices, temperature=0.07):
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature
    return F.cross_entropy(sim_matrix, positive_indices)


def compute_bbox_loss(pred_bbox, target_bbox):
    """
    Compute bbox loss: L1 loss + IoU loss.

    Args:
        pred_bbox: Predicted bbox [B, 4] normalized [0, 1]
        target_bbox: Target bbox [B, 4] normalized [0, 1]

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

    return 0.5 * l1_loss + 0.5 * iou_loss.mean()


def main(save_path):
    not_update = 0
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
    processor_sat = AutoImageProcessor.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
    )

    model = Encoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    img_to_text_dict = json.load(open(TEXT_FILE, "r"))
    bbox_dict = json.load(open(BBOX_FILE, "r"))

    print("Setting up dataset and dataloader...")
    # Load training image pairs from /data/feihong/ckpt/train.txt
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
    train_dataset = TargetSearchDataset(
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

    test_dataset = TargetSearchDataset(
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
    min_loss = 100
    BBOX_LOSS_WEIGHT = 0.1

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_bbox_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for i, batch in enumerate(progress_bar):
            target_pixel_values = batch["target_pixel_values"].to(DEVICE)
            search_pixel_values = batch["search_pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            local_indices = batch["index"].to(DEVICE)
            target_bbox = batch["bbox"].to(DEVICE)

            bbox_pred, anchor_feats, grid_feats = model.bbox_forward(
                target_pixel_values, search_pixel_values
            )
            candidate_feats = grid_feats.reshape(-1, PROJECTION_DIM)
            text_feats = model.text_forward(input_ids)

            positive_indices = torch.zeros(
                local_indices.shape[0], local_indices.shape[0] * 9, device=DEVICE
            )
            batch_offsets = torch.arange(local_indices.shape[0], device=DEVICE) * 9
            row_indices_broad = torch.arange(
                local_indices.shape[0], device=DEVICE
            ).unsqueeze(1)
            col_offsets = torch.arange(9, device=DEVICE).unsqueeze(0)
            same_image_cols = batch_offsets.unsqueeze(1) + col_offsets

            positive_indices[row_indices_broad, same_image_cols] = 0.5
            global_positive_indices = local_indices + batch_offsets
            row_indices_flat = torch.arange(local_indices.shape[0], device=DEVICE)
            positive_indices[row_indices_flat, global_positive_indices] = 0.95

            img_text_loss = info_nce_loss(
                anchor_feats, candidate_feats, positive_indices
            ) + info_nce_loss(text_feats, candidate_feats, positive_indices)

            bbox_loss = compute_bbox_loss(bbox_pred, target_bbox)

            loss = img_text_loss + BBOX_LOSS_WEIGHT * bbox_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()
            progress_bar.set_postfix(
                {
                    "img_text_loss": f"{img_text_loss.item():.4f}",
                    "bbox_loss": f"{bbox_loss.item():.4f}",
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
        writer.add_scalar("Loss/val_img_epoch", val_metrics["img_loss"], epoch)
        writer.add_scalar("Loss/val_txt_epoch", val_metrics["txt_loss"], epoch)
        writer.add_scalar("Loss/val_bbox_epoch", val_metrics["bbox_loss"], epoch)
        print(
            f"Epoch {epoch + 1} finished. "
            f"Avg train Loss: {avg_loss:.4f} (bbox: {avg_bbox_loss:.4f}). "
            f"Val txt loss: {val_metrics['txt_loss']:.4f}, bbox loss: {val_metrics['bbox_loss']:.4f}"
        )

        if val_metrics["txt_loss"] < min_loss:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/best.pth")
            min_loss = val_metrics["txt_loss"]
            not_update = 0
        else:
            not_update += 1
        if not_update > 5:
            print("Validation loss not improving. Stopping early.")
            break

    print("Training complete.")
    writer.close()


def compute_iou(pred_bbox, gt_bbox):
    """
    Compute Intersection over Union (IoU) between predicted and ground truth bboxes.

    Args:
        pred_bbox: Predicted bbox [4] normalized [0, 1]
        gt_bbox: Target bbox [4] normalized [0, 1]

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


def validation(model, loader, epoch):
    progress_bar = tqdm(loader, desc=f"Valid {epoch + 1}/{NUM_EPOCHS}")
    total_img_loss = 0
    total_txt_loss = 0
    total_bbox_loss = 0

    for batch in progress_bar:
        target_pixel_values = batch["target_pixel_values"].to(DEVICE)
        search_pixel_values = batch["search_pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        local_indices = batch["index"].to(DEVICE)
        target_bbox = batch["bbox"].to(DEVICE)

        batch_offsets = torch.arange(local_indices.shape[0], device=DEVICE) * 9
        positive_indices = local_indices + batch_offsets

        with torch.no_grad():
            bbox_pred, anchor_feats, grid_feats = model.bbox_forward(
                target_pixel_values, search_pixel_values
            )
            text_feats = model.text_forward(input_ids)

            candidate_feats = grid_feats.reshape(-1, PROJECTION_DIM)

            img_loss = info_nce_loss(anchor_feats, candidate_feats, positive_indices)
            txt_loss = info_nce_loss(
                text_feats + anchor_feats, candidate_feats, positive_indices
            )
            bbox_loss = compute_bbox_loss(bbox_pred, target_bbox)

            total_img_loss += img_loss.item()
            total_txt_loss += txt_loss.item()
            total_bbox_loss += bbox_loss.item()

    avg_img_loss = total_img_loss / len(loader)
    avg_txt_loss = total_txt_loss / len(loader)
    avg_bbox_loss = total_bbox_loss / len(loader)

    return {
        "img_loss": avg_img_loss,
        "txt_loss": avg_txt_loss,
        "bbox_loss": avg_bbox_loss,
    }


def eval(run=False):
    def calcu_cos(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b_norm, a_norm[..., None]).flatten()

    if run:
        img_to_text_dict = json.load(open(TEXT_FILE, "r"))
        bbox_dict = json.load(open(BBOX_FILE, "r"))

        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        processor_sat = AutoImageProcessor.from_pretrained(
            MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        encoder = Encoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
        model_path = f"../ckpt/unified_siglip/best.pth"

        if os.path.exists(model_path):
            encoder.load_state_dict(
                torch.load(model_path, map_location="cpu"), strict=False
            )
        encoder.eval()

        res_search = {}
        res_fused_query = {}
        res_bbox_pred = {}
        res_bbox_gt = {}

        print("Setting up dataset and dataloader for eval...")
        eval_list = []
        with open("/data/feihong/ckpt/test.txt", "r") as f:
            for line in f:
                query_path = line.strip()
                eval_list.append(query_path)

        for img_path in tqdm(eval_list):
            name = img_path.split("/")[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            if os.path.exists(search_path):
                try:
                    query_image = Image.open(img_path).convert("RGB")
                    search_image = Image.open(search_path).convert("RGB")
                    search_image = search_image.crop((840, 0, 3000, 2160)).resize(
                        (640, 640)
                    )

                    text_name = name + "_01"
                    text_description = img_to_text_dict.get(text_name, "")
                    for noun in [
                        "**",
                        "\n",
                        "noun",
                        "phrases",
                        "Phrase",
                        "Noun",
                        "Summary",
                        "Environment",
                        "32 tokens",
                    ]:
                        text_description = text_description.replace(noun, "")

                    bbox_key = f"{name}.png"
                    gt_bbox = bbox_dict.get(bbox_key, None)
                    if gt_bbox is None:
                        gt_bbox = bbox_dict.get(bbox_key.replace(".png", ".jpeg"), None)
                    if gt_bbox is None:
                        gt_bbox = (1536, 656, 2268, 1374)

                    query_inputs = processor(images=query_image, return_tensors="pt")[
                        "pixel_values"
                    ].to(DEVICE)
                    search_inputs = processor_sat(
                        images=search_image, return_tensors="pt"
                    )["pixel_values"].to(DEVICE)
                    text_inputs = tokenizer(
                        text_description,
                        padding="max_length",
                        truncation=True,
                        max_length=64,
                        return_tensors="pt",
                    )
                    input_ids = text_inputs["input_ids"].to(DEVICE)

                    with torch.no_grad():
                        bbox_pred, anchor_feats, grid_feats = encoder.bbox_forward(
                            query_inputs, search_inputs
                        )
                        text_feats = encoder.text_forward(input_ids)
                        fused_query_feats = anchor_feats + text_feats

                    res_search[name] = grid_feats.cpu().numpy()
                    res_fused_query[name] = fused_query_feats.cpu().numpy()
                    res_bbox_pred[name] = bbox_pred.cpu().numpy()
                    res_bbox_gt[name] = np.array(gt_bbox)
                except Exception as e:
                    print(f"Error processing {name}: {e}")

        np.savez("eval_search_siglip.npz", **res_search)
        np.savez("eval_fused_query_siglip.npz", **res_fused_query)
        np.savez("eval_bbox_pred.npz", **res_bbox_pred)
        np.savez("eval_bbox_gt.npz", **res_bbox_gt)
        print("Evaluation feature extraction complete.")
    else:
        search_res = np.load("eval_search_siglip.npz")
        fused_query_res = np.load("eval_fused_query_siglip.npz")
        bbox_pred_res = np.load("eval_bbox_pred.npz")
        bbox_gt_res = np.load("eval_bbox_gt.npz")

        distances = json.load(open("/data/feihong/ckpt/distances.json", "r"))
        test_num = 100
        test_list = [k for k in search_res.keys()]
        res = {}
        top1 = 0
        top5 = 0
        top10 = 0

        all_ious = []

        if not test_list:
            print("No evaluation data found in npz files.")
            return

        for key in tqdm(test_list):
            fused_query_feature = fused_query_res[key]

            ex_img_list = random.sample(
                test_list, min(test_num - 1, len(test_list) - 1)
            )
            if key in ex_img_list:
                ex_img_list.remove(key)
            ex_img_list.append(key)

            candidate_indices = [len(ex_img_list) - 1]
            for i, name in enumerate(ex_img_list[:-1]):
                if (
                    f"{name}.kml" not in distances
                    or f"{key}.kml" not in distances[f"{name}.kml"]
                ):
                    continue
                if distances[f"{name}.kml"][f"{key}.kml"] < 380:
                    candidate_indices.append(i)

            res[key] = []
            for img_name in ex_img_list:
                cos_sim_grid = calcu_cos(fused_query_feature, search_res[img_name])
                res[key].append(cos_sim_grid)

            img_res = np.array(res[key]).mean(1).argsort()[-15:][::-1]

            if any(cand in candidate_indices for cand in img_res[:1]):
                top1 += 1
            if any(cand in candidate_indices for cand in img_res[:5]):
                top5 += 1
            if any(cand in candidate_indices for cand in img_res[:10]):
                top10 += 1

            if key in bbox_pred_res and key in bbox_gt_res:
                pred_bbox = bbox_pred_res[key]
                gt_bbox = bbox_gt_res[key]

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

                if union_area > 0:
                    iou = inter_area / union_area
                else:
                    iou = 0.0
                all_ious.append(iou)

        mean_iou = np.mean(all_ious) if all_ious else 0.0
        iou_at_05 = (
            sum(1 for iou in all_ious if iou >= 0.5) / len(all_ious)
            if all_ious
            else 0.0
        )

        print(f"Retrieval Metrics:")
        print(f"Top 1: {top1} / {len(test_list)} ({top1 / len(test_list) * 100:.2f}%)")
        print(f"Top 5: {top5} / {len(test_list)} ({top5 / len(test_list) * 100:.2f}%)")
        print(
            f"Top 10: {top10} / {len(test_list)} ({top10 / len(test_list) * 100:.2f}%)"
        )
        print(f"\nBbox Metrics:")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"IoU@0.5: {iou_at_05:.4f}")


def eval_denseuav(run=False):
    DENSE_UAV_ROOT = "/data/feihong/DenseUAV"
    TEST_DENSE_FILE = "/data/feihong/ckpt/test_dense.txt"

    def calcu_cos(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b_norm, a_norm[..., None]).flatten()

    if run:
        img_to_text_dict = json.load(
            open("/data/feihong/drone_text_single_long.json", "r")
        )
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        encoder = Encoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
        model_path = f"../ckpt/train_siglip_proj/best.pth"

        if os.path.exists(model_path):
            encoder.load_state_dict(
                torch.load(model_path, map_location="cpu"), strict=False
            )
        encoder.eval()

        res_search = {}
        res_fused_query = {}

        print("Setting up dataset and dataloader for eval_denseuav...")
        eval_list = []
        with open(TEST_DENSE_FILE, "r") as f:
            for line in f:
                query_path = line.strip()
                eval_list.append(query_path)

        for img_path in tqdm(eval_list):
            parts = img_path.split("/")
            drone_id = parts[-2]
            is_train = "train" in parts

            if is_train:
                search_path = f"{DENSE_UAV_ROOT}/train/satellite/{drone_id}/H100.tif"
            else:
                search_path = (
                    f"{DENSE_UAV_ROOT}/test/gallery_satellite/{drone_id}/H100.tif"
                )

            if os.path.exists(search_path):
                try:
                    query_image = Image.open(img_path).convert("RGB")
                    search_image = Image.open(search_path).convert("RGB")
                    search_image = search_image.resize((640, 640))

                    text_name = drone_id + "_01"
                    text_description = img_to_text_dict.get(text_name, "")
                    for noun in [
                        "**",
                        "\n",
                        "noun",
                        "phrases",
                        "Phrase",
                        "Noun",
                        "Summary",
                        "Environment",
                        "32 tokens",
                    ]:
                        text_description = text_description.replace(noun, "")

                    query_inputs = processor(images=query_image, return_tensors="pt")[
                        "pixel_values"
                    ].to(DEVICE)
                    search_inputs = processor(images=search_image, return_tensors="pt")[
                        "pixel_values"
                    ].to(DEVICE)
                    text_inputs = tokenizer(
                        text_description,
                        padding="max_length",
                        truncation=True,
                        max_length=64,
                        return_tensors="pt",
                    )
                    input_ids = text_inputs["input_ids"].to(DEVICE)

                    with torch.no_grad():
                        anchor_feats = encoder.query_forward(query_inputs)
                        grid_feats = encoder.sat_forward(search_inputs)
                        # text_feats = encoder.text_forward(input_ids)
                        fused_query_feats = anchor_feats

                    res_search[drone_id] = grid_feats.cpu().numpy()
                    res_fused_query[drone_id] = fused_query_feats.cpu().numpy()
                except Exception as e:
                    print(f"Error processing {drone_id}: {e}")

        np.savez("eval_denseuav_search.npz", **res_search)
        np.savez("eval_denseuav_fused_query.npz", **res_fused_query)
        print("Evaluation feature extraction complete.")
    else:
        search_res = np.load("eval_denseuav_search.npz")
        fused_query_res = np.load("eval_denseuav_fused_query.npz")

        test_list = [k for k in search_res.keys()]
        res = {}
        top1 = 0
        top5 = 0
        top10 = 0

        if not test_list:
            print("No evaluation data found in npz files.")
            return

        for key in tqdm(test_list):
            fused_query_feature = fused_query_res[key]

            ex_img_list = random.sample(test_list, min(99, len(test_list) - 1))
            if key in ex_img_list:
                ex_img_list.remove(key)
            ex_img_list.append(key)
            right_num = len(ex_img_list) - 1
            res[key] = []
            for img_name in ex_img_list:
                cos_sim_grid = calcu_cos(fused_query_feature, search_res[img_name])
                res[key].append(cos_sim_grid)

            img_res = np.array(res[key]).mean(1).argsort()[-15:][::-1]

            if right_num in img_res[:1]:
                top1 += 1
            if right_num in img_res[:5]:
                top5 += 1
            if right_num in img_res[:10]:
                top10 += 1

        print(f"Top 1: {top1} / {len(test_list)} ({top1 / len(test_list) * 100:.2f}%)")
        print(f"Top 5: {top5} / {len(test_list)} ({top5 / len(test_list) * 100:.2f}%)")
        print(
            f"Top 10: {top10} / {len(test_list)} ({top10 / len(test_list) * 100:.2f}%)"
        )


if __name__ == "__main__":
    exp_name = "unified_siglip"
    save_dir = f"../ckpt/{exp_name}"

    if os.path.exists(save_dir):
        print(f"Experiment directory '{save_dir}' already exists.")
        print("Re-using directory. Logs and checkpoints might be overwritten.")
    else:
        os.makedirs(save_dir)
        print(f"Created experiment directory: {save_dir}")

    # Run training
    main(save_dir)

    # Run evaluation
    # eval(True)  # Extract features
    # eval(False) # Calculate metrics

    # eval_denseuav(True)  # Extract features
    # eval_denseuav(False)  # Extract features
