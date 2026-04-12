import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
from tqdm import tqdm
import os
import sys
from glob import glob
from pathlib import Path
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import ShiftedSatelliteDroneDataset

SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# --- Configuration ---
MODEL_NAME = "google/siglip-base-patch16-224"
CACHE_DIR = "/data/feihong/hf_cache"
DRONE_VIEW_FOLDER = "/data/feihong/drone_view"
IMAGE_FOLDER = "/data/feihong/image_1024"
HEADING_FOLDER = "/data/feihong/range_250"

NUM_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECTION_DIM = 768
DRONE_SIZE = (256, 256)
NUM_WORKERS = min(16, os.cpu_count() or 8)
PREFETCH_FACTOR = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PROCESSOR_IMAGE_SIZE = {"height": 432, "width": 768}
DEFAULT_SUBSET_HEIGHTS = [250, 300]
DEFAULT_SUBSET_ANGLES = [0, 90, 180, 270]
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _set_processor_size(processor, size):
    # For non-native resolutions, we rely on interpolate_pos_encoding in forward.
    if size is None:
        return
    if hasattr(processor, "image_processor"):
        processor.image_processor.size = dict(size)
    elif hasattr(processor, "size"):
        processor.size = dict(size)


def _get_patch_size(vision_model):
    patch_size = getattr(vision_model.config, "patch_size", 16)
    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) == 2:
            return int(patch_size[0]), int(patch_size[1])
        return int(patch_size[0]), int(patch_size[0])
    patch_size = int(patch_size)
    return patch_size, patch_size


def _normalize_subset(values, defaults):
    if values is None:
        return {int(v) for v in defaults}
    if isinstance(values, (list, tuple, set)):
        return {int(v) for v in values}
    return {int(values)}


class TargetSearchDataset(Dataset):
    def __init__(
        self, image_pairs, processor, tokenizer, img_to_text_dict, mode="train"
    ):
        self.image_paths = image_pairs
        self.processor = processor
        self.tokenizer = tokenizer
        self.img_to_text_dict = img_to_text_dict

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

        img_id = query_path.split("/")[-2]

        choice = []
        for number in ["01.", "21.", "31.", "41.", "51."]:
            new_query_path = query_path.replace("01.", number)
            if not os.path.exists(new_query_path):
                continue
            choice.append(new_query_path)

        heading0_path = f"{HEADING_FOLDER}/{img_id}_range250_heading0.png"
        if os.path.exists(heading0_path):
            choice.append(heading0_path)

        if choice:
            query_path = random.sample(choice, 1)[0]

        if "_range250_heading0" in query_path:
            name = query_path.split("/")[-1].split("_")[0]
        else:
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

        if "_range250_heading0" in query_path:
            query_image = query_image.crop((420, 0, 1500, 1080))

        if self.mode == "train":
            width, height = search_image.size

            min_left = (width / 2) - self.crop_size
            max_left = width / 2
            min_top = (height / 2) - self.crop_size
            max_top = height / 2

            min_left = max(0, min_left)
            min_top = max(0, min_top)
            max_left = min(width - self.crop_size, max_left)
            max_top = min(height - self.crop_size, max_top)

            if max_left < min_left:
                max_left = min_left
            if max_top < min_top:
                max_top = min_top

            left = random.uniform(min_left, max_left)
            top = random.uniform(min_top, max_top)

            right = left + self.crop_size
            bottom = top + self.crop_size

            augmented_crop_image = search_image.crop((left, top, right, bottom))
            augmented_crop_image = augmented_crop_image.resize((640, 640))

            center_x_in_crop = (width / 2) - left
            center_y_in_crop = (height / 2) - top

            normalized_center_pos = torch.tensor(
                [center_x_in_crop / self.crop_size, center_y_in_crop / self.crop_size],
                dtype=torch.float32,
            )

            col_idx = min(int(normalized_center_pos[0] * 3), 2)
            row_idx = min(int(normalized_center_pos[1] * 3), 2)
            index = row_idx * 3 + col_idx
        else:
            augmented_crop_image = search_image.crop((840, 0, 3000, 2160)).resize(
                (640, 640)
            )
            index = 4

        text_inputs = self.tokenizer(
            [text_description],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        query_inputs = self.processor(query_image)
        search_inputs = self.processor(augmented_crop_image)

        return {
            "target_pixel_values": query_inputs["pixel_values"][0],
            "search_pixel_values": search_inputs["pixel_values"][0],
            "input_ids": text_inputs["input_ids"][0],
            # "attention_mask": text_inputs["attention_mask"][0],
            "index": index,
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

        self.pool = nn.AdaptiveAvgPool2d((3, 3))

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.text_feature_dim * 2, proj_dim),
        )

    def ref_forward(self, pixel_values):
        vision_outputs = self.vision_model(
            pixel_values, interpolate_pos_encoding=True
        )
        vision_features = vision_outputs.last_hidden_state

        B, N, D = vision_features.shape
        patch_h, patch_w = _get_patch_size(self.vision_model)
        grid_h = pixel_values.shape[-2] // max(1, patch_h)
        grid_w = pixel_values.shape[-1] // max(1, patch_w)

        if N == grid_h * grid_w:
            patch_tokens = vision_features
        elif N - 1 == grid_h * grid_w:
            patch_tokens = vision_features[:, 1:, :]
        else:
            # Fallback for unexpected token layouts.
            side = int((N) ** 0.5)
            if side * side == N:
                grid_h = side
                grid_w = side
                patch_tokens = vision_features
            elif side * side == (N - 1):
                grid_h = side
                grid_w = side
                patch_tokens = vision_features[:, 1:, :]
            else:
                raise RuntimeError(
                    "Unable to infer patch grid size from SigLIP features: "
                    f"N={N}, expected {grid_h * grid_w} or {grid_h * grid_w + 1}."
                )

        patch_tokens_for_pooling = patch_tokens.permute(0, 2, 1).reshape(
            B, D, grid_h, grid_w
        )
        pooled_features = self.pool(patch_tokens_for_pooling)
        final_features = pooled_features.flatten(2).permute(0, 2, 1)

        return final_features

    def query_forward(self, pixel_values):
        pooled_features = self.vision_model(
            pixel_values, interpolate_pos_encoding=True
        ).pooler_output
        return pooled_features

    def text_forward(self, input_ids, attention_mask=None):
        text_outputs = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooler_output = text_outputs.pooler_output
        proj_feature = self.text_projector(pooler_output)
        return proj_feature


def info_nce_loss(query_feats, candidate_feats, positive_indices, temperature=0.07):
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature
    return F.cross_entropy(sim_matrix, positive_indices)


def main(save_path):
    not_update = 0
    exp_name = save_path.split("/")[-1] if save_path else "default_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Loading models and processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor_sat = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    _set_processor_size(processor_sat, PROCESSOR_IMAGE_SIZE)

    model = Encoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    use_cuda = DEVICE.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    print("Setting up dataset and dataloader...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="train",
        subset_heights=DEFAULT_SUBSET_HEIGHTS,
        subset_angles=DEFAULT_SUBSET_ANGLES,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and use_cuda,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        drop_last=True,
    )

    test_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="val",
        subset_heights=DEFAULT_SUBSET_HEIGHTS,
        subset_angles=DEFAULT_SUBSET_ANGLES,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and use_cuda,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")
    min_loss = 100
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for i, batch in enumerate(progress_bar):
            target_pixel_values = batch["target_pixel_values"].to(
                DEVICE, non_blocking=True
            )
            search_pixel_values = batch["search_pixel_values"].to(
                DEVICE, non_blocking=True
            )
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            local_indices = batch["index"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=False):
                anchor_feats = model.query_forward(target_pixel_values)
                grid_feats = model.ref_forward(search_pixel_values)
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

                loss = img_text_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({"img_text_loss": f"{img_text_loss.item():.4f}"})
            writer.add_scalar(
                "Loss/train_batch", loss.item(), epoch * len(train_dataloader) + i
            )

        avg_loss = total_loss / len(train_dataloader)

        model.eval()
        val_loss, txt_val_loss = validation(model, test_dataloader, epoch)
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Loss/val_img_epoch", val_loss, epoch)
        writer.add_scalar("Loss/val_txt_epoch", txt_val_loss, epoch)
        print(
            f"Epoch {epoch + 1} finished. Average train Loss: {avg_loss:.4f}. Average test Loss: {txt_val_loss:.4f}"
        )

        if txt_val_loss < min_loss:
            os.makedirs(save_path, exist_ok=True)
            # Save full model state dict (including projection heads)
            torch.save(model.state_dict(), f"{save_path}/best.pth")
            min_loss = txt_val_loss
            not_update = 0
        else:
            not_update += 1
        if not_update > 5:
            print("Validation loss not improving. Stopping early.")
            break

    print("Training complete.")
    writer.close()


def validation(model, loader, epoch):
    progress_bar = tqdm(loader, desc=f"Valid {epoch + 1}/{NUM_EPOCHS}")
    total_loss = 0
    txt_total = 0
    use_cuda = DEVICE.startswith("cuda")
    for batch in progress_bar:
        target_pixel_values = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        search_pixel_values = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        local_indices = batch["index"].to(DEVICE, non_blocking=True)

        batch_offsets = torch.arange(local_indices.shape[0], device=DEVICE) * 9
        positive_indices = local_indices + batch_offsets

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            anchor_feats = model.query_forward(target_pixel_values)
            grid_feats = model.ref_forward(search_pixel_values)
            text_feats = model.text_forward(input_ids)

            candidate_feats = grid_feats.reshape(-1, PROJECTION_DIM)

            img_txt_loss = info_nce_loss(
                text_feats + anchor_feats, candidate_feats, positive_indices
            )

            total_loss += img_txt_loss.item()
            txt_total += img_txt_loss.item()

    avg_loss = total_loss / len(loader)
    txt_avg_loss = txt_total / len(loader)
    return avg_loss, txt_avg_loss


def extract_eval_features(
    checkpoint_path=None,
    query_feat_path="eval_siglip_query_feats.pt",
    gallery_feat_path="eval_siglip_gallery_feats.pt",
    subset_heights=None,
    subset_angles=None,
):
    print("Stage 1/2: extracting and saving query/gallery features...")
    use_cuda = DEVICE.startswith("cuda")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor_sat = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _set_processor_size(processor_sat, PROCESSOR_IMAGE_SIZE)

    subset_heights = _normalize_subset(subset_heights, DEFAULT_SUBSET_HEIGHTS)
    subset_angles = _normalize_subset(subset_angles, DEFAULT_SUBSET_ANGLES)

    test_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="test",
        subset_heights=sorted(subset_heights),
        subset_angles=sorted(subset_angles),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and use_cuda,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )

    encoder = Encoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
    model_path = checkpoint_path or "../ckpt/retrieval_siglip/best.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    encoder.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    encoder.eval()

    query_feats_list = []
    query_labels = []
    query_heights = []
    query_angles = []
    gallery_feat_dict = {}

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Extracting query and gallery features"):
            query_inputs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
            search_inputs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            satellite_paths = batch["satellite_path"]

            with torch.amp.autocast("cuda", enabled=False):
                anchor_feats = encoder.query_forward(query_inputs)
                text_feats = encoder.text_forward(input_ids)
                fused_query_feats = F.normalize(anchor_feats + text_feats, p=2, dim=1)
            query_feats_list.append(fused_query_feats.cpu())

            batch_labels = [
                os.path.splitext(os.path.basename(path))[0] for path in satellite_paths
            ]
            query_labels.extend(batch_labels)
            query_heights.extend(batch["height"].cpu().tolist())
            query_angles.extend(batch["angle"].cpu().tolist())

            unseen_indices = []
            unseen_labels = []
            for i, label in enumerate(batch_labels):
                if label not in gallery_feat_dict:
                    unseen_indices.append(i)
                    unseen_labels.append(label)

            if unseen_indices:
                gallery_batch = search_inputs[unseen_indices]
                with torch.amp.autocast("cuda", enabled=False):
                    gallery_grid = encoder.ref_forward(gallery_batch)
                    gallery_grid = F.normalize(gallery_grid, p=2, dim=2)
                gallery_grid = gallery_grid.cpu()
                for i, label in enumerate(unseen_labels):
                    gallery_feat_dict[label] = gallery_grid[i]

    if not query_feats_list or not gallery_feat_dict:
        print("No valid features extracted.")
        return

    query_payload = {
        "features": torch.cat(query_feats_list, dim=0),
        "labels": query_labels,
        "heights": query_heights,
        "angles": query_angles,
    }

    gallery_labels = sorted(gallery_feat_dict.keys())
    gallery_feats = torch.stack(
        [gallery_feat_dict[label] for label in gallery_labels], dim=0
    )
    gallery_payload = {
        "features": gallery_feats,
        "labels": gallery_labels,
    }

    torch.save(query_payload, query_feat_path)
    torch.save(gallery_payload, gallery_feat_path)
    print(f"Saved query features to: {query_feat_path}")
    print(f"Saved gallery features to: {gallery_feat_path}")
    print(f"Queries: {len(query_labels)}, Gallery: {len(gallery_labels)}")


def score_eval_from_saved(
    query_feat_path="eval_siglip_query_feats.pt",
    gallery_feat_path="eval_siglip_gallery_feats.pt",
    subset_heights=None,
    subset_angles=None,
    test_num=100,
):
    print("Stage 2/2: loading saved features and scoring...")
    if not os.path.exists(query_feat_path):
        raise FileNotFoundError(f"Query feature file not found: {query_feat_path}")
    if not os.path.exists(gallery_feat_path):
        raise FileNotFoundError(f"Gallery feature file not found: {gallery_feat_path}")

    query_payload = torch.load(query_feat_path, map_location="cpu")
    gallery_payload = torch.load(gallery_feat_path, map_location="cpu")

    include_file = "/data/feihong/ckpt/include.json"
    include_map = {}

    def _norm_label(label):
        return str(label).strip().split(".")[0]

    if os.path.exists(include_file):
        with open(include_file, "r") as f:
            include_raw = json.load(f)

        for key, values in include_raw.items():
            norm_key = _norm_label(key)
            if isinstance(values, list):
                include_map[norm_key] = {_norm_label(v) for v in values}
            else:
                include_map[norm_key] = {_norm_label(values)}
    else:
        print(f"Warning: include file not found: {include_file}. Fallback to exact-match GT.")

    query_feats = query_payload["features"]
    query_labels = query_payload["labels"]
    query_heights = query_payload["heights"]
    query_angles = query_payload["angles"]
    subset_heights = _normalize_subset(subset_heights, DEFAULT_SUBSET_HEIGHTS)
    subset_angles = _normalize_subset(subset_angles, DEFAULT_SUBSET_ANGLES)

    gallery_feats = gallery_payload["features"]
    gallery_labels = gallery_payload["labels"]

    selected_indices = []
    for idx, (h, a) in enumerate(zip(query_heights, query_angles)):
        if int(h) not in subset_heights:
            continue
        if int(a) not in subset_angles:
            continue
        selected_indices.append(idx)

    if not selected_indices:
        print(
            f"No queries found for filters: heights={sorted(subset_heights)}, angles={sorted(subset_angles)}."
        )
        return

    query_feats = query_feats[selected_indices]
    query_labels = [query_labels[idx] for idx in selected_indices]
    print(
        f"Scoring {len(selected_indices)} / {len(query_payload['labels'])} queries "
        f"(heights={sorted(subset_heights)}, angles={sorted(subset_angles)}, test_num={test_num})"
    )

    if test_num is None:
        test_num = len(gallery_labels)
    test_num = int(test_num)
    if test_num <= 0:
        raise ValueError("test_num must be a positive integer.")

    query_feats = query_feats.to(DEVICE)
    gallery_feats = gallery_feats.to(DEVICE)
    label_to_gallery_index = {label: idx for idx, label in enumerate(gallery_labels)}
    norm_gallery_labels = [_norm_label(label) for label in gallery_labels]

    valid_queries = 0
    top1 = 0
    top5 = 0
    top10 = 0

    for q_idx, gt_label in enumerate(query_labels):
        gt_gallery_index = label_to_gallery_index.get(gt_label)
        if gt_gallery_index is None:
            continue

        gt_norm = _norm_label(gt_label)
        positive_label_set = set(include_map.get(gt_norm, set()))
        positive_label_set.add(gt_norm)

        all_indices = list(range(len(gallery_labels)))
        if test_num >= len(all_indices):
            candidate_indices = all_indices
        else:
            negative_pool = [idx for idx in all_indices if idx != gt_gallery_index]
            sampled_negatives = random.sample(negative_pool, test_num - 1)
            candidate_indices = sampled_negatives + [gt_gallery_index]
            random.shuffle(candidate_indices)

        candidate_gallery_feats = gallery_feats[candidate_indices]
        score_vec = torch.einsum(
            "d,knd->kn", query_feats[q_idx], candidate_gallery_feats
        ).mean(dim=1)

        k_max = min(10, score_vec.shape[0])
        topk_local_indices = torch.topk(score_vec, k=k_max, dim=0).indices.tolist()

        positive_local_indices = [
            local_idx
            for local_idx, global_idx in enumerate(candidate_indices)
            if norm_gallery_labels[global_idx] in positive_label_set
        ]

        valid_queries += 1
        if any(idx in topk_local_indices[:1] for idx in positive_local_indices):
            top1 += 1
        if any(idx in topk_local_indices[: min(5, k_max)] for idx in positive_local_indices):
            top5 += 1
        if any(
            idx in topk_local_indices[: min(10, k_max)]
            for idx in positive_local_indices
        ):
            top10 += 1

    if valid_queries == 0:
        print("No valid queries matched gallery labels.")
        return

    print(f"Top 1: {top1} / {valid_queries} ({top1 / valid_queries * 100:.2f}%)")
    print(f"Top 5: {top5} / {valid_queries} ({top5 / valid_queries * 100:.2f}%)")
    print(f"Top 10: {top10} / {valid_queries} ({top10 / valid_queries * 100:.2f}%)")


def eval(
    checkpoint_path=None,
    subset_heights=None,
    subset_angles=None,
    test_num=100,
    query_feat_path="eval_siglip_query_feats.pt",
    gallery_feat_path="eval_siglip_gallery_feats.pt",
    run_extract=True,
):
    subset_heights = _normalize_subset(subset_heights, DEFAULT_SUBSET_HEIGHTS)
    subset_angles = _normalize_subset(subset_angles, DEFAULT_SUBSET_ANGLES)

    if run_extract:
        extract_eval_features(
            checkpoint_path=checkpoint_path,
            query_feat_path=query_feat_path,
            gallery_feat_path=gallery_feat_path,
            subset_heights=subset_heights,
            subset_angles=subset_angles,
        )
    score_eval_from_saved(
        query_feat_path=query_feat_path,
        gallery_feat_path=gallery_feat_path,
        subset_heights=subset_heights,
        subset_angles=subset_angles,
        test_num=test_num,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SigLIP retrieval train/eval")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run training loop.",
    )
    args = parser.parse_args()

    exp_name = "retrieval_siglip"
    save_dir = f"../ckpt/{exp_name}"

    if os.path.exists(save_dir):
        print(f"Experiment directory '{save_dir}' already exists.")
        print("Re-using directory. Logs and checkpoints might be overwritten.")
    else:
        os.makedirs(save_dir)
        print(f"Created experiment directory: {save_dir}")

    # Run training
    if not args.eval:
        main(save_dir)
    else:
        # Run evaluation
        eval(
            f"{save_dir}/best.pth",
            subset_heights=DEFAULT_SUBSET_HEIGHTS,
            subset_angles=DEFAULT_SUBSET_ANGLES,
        )  # Extract features + score Top-1/5/10
