import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open_clip
from PIL import Image
from tqdm import tqdm
import os
import sys
import time
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
MODEL_NAME = "hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k"
CACHE_DIR = "/data/feihong/hf_cache"
DRONE_VIEW_FOLDER = "/data/feihong/drone_view"
IMAGE_FOLDER = "/data/feihong/image_1024"
HEADING_FOLDER = "/data/feihong/range_250"

NUM_EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
PROJECTION_DIM = 768
NUM_WORKERS = min(16, os.cpu_count() or 8)
PREFETCH_FACTOR = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
ENABLE_DATASET_TIMING_LOG = False
DATASET_TIMING_LOG_INTERVAL = 200
ENABLE_BATCH_TIMING_LOG = False
BATCH_TIMING_LOG_INTERVAL = 50
ENABLE_SEARCH_PIXEL_VIS = True
SEARCH_PIXEL_VIS_INTERVAL = 200
SEARCH_PIXEL_VIS_MAX_IMAGES = 8


class OpenClipImageProcessorWrapper:
    def __init__(self, preprocess):
        self.preprocess = preprocess
        self.size = {"height": 432, "width": 768}

    def __call__(self, images, return_tensors="pt"):
        pixel_values = self.preprocess(images)
        return {"pixel_values": pixel_values.unsqueeze(0)}


class OpenClipTokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self,
        text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ):
        input_ids = self.tokenizer([text])
        return {"input_ids": input_ids}


def _to_uint8_pil_from_pixel_tensor(pixel_tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized CHW tensor to a viewable RGB image."""
    img = pixel_tensor.detach().cpu().float()
    if img.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(img.shape)}")

    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] != 3:
        raise ValueError(f"Expected channel size 1 or 3, got {img.shape[0]}")

    # Most CLIP/EVA preprocess pipelines normalize with these stats.
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    if img.min().item() < 0.0 or img.max().item() > 1.0:
        img = img * clip_std + clip_mean

    img = img.clamp(0.0, 1.0)
    img_np = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(img_np)


def visualize_search_pixel_values(
    search_pixel_values: torch.Tensor,
    save_path: str,
    max_images: int = 8,
    ncols: int = 4,
    padding: int = 8,
) -> str:
    """Save a grid visualization for search_pixel_values ([B, C, H, W] or [C, H, W])."""
    if search_pixel_values.ndim == 3:
        batch = search_pixel_values.unsqueeze(0)
    elif search_pixel_values.ndim == 4:
        batch = search_pixel_values
    else:
        raise ValueError(
            f"Expected tensor dims 3 or 4, got ndim={search_pixel_values.ndim}"
        )

    num_images = min(int(batch.shape[0]), int(max_images))
    if num_images <= 0:
        raise ValueError("No images to visualize.")

    pil_images = [_to_uint8_pil_from_pixel_tensor(batch[i]) for i in range(num_images)]
    img_w, img_h = pil_images[0].size

    cols = max(1, min(int(ncols), num_images))
    rows = (num_images + cols - 1) // cols

    canvas_w = cols * img_w + (cols + 1) * padding
    canvas_h = rows * img_h + (rows + 1) * padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))

    for i, img in enumerate(pil_images):
        r = i // cols
        c = i % cols
        x = padding + c * (img_w + padding)
        y = padding + r * (img_h + padding)
        canvas.paste(img, (x, y))

    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    canvas.save(save_path)
    return save_path


class TargetSearchDataset(Dataset):
    def __init__(
        self, image_pairs, preprocess, tokenizer, img_to_text_dict, mode="train"
    ):
        self.image_paths = image_pairs
        self.preprocess = preprocess
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
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        query_path, search_path = self.image_paths[idx]

        img_id = query_path.split("/")[-2]

        choice = []
        for number in ["01", "21", "31", "41", "51"]:
            new_query_path = query_path.replace("01", number)
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

        # Process images and text with EVA-CLIP preprocessors
        query_image_tensor = self.preprocess(query_image)
        search_image_tensor = self.preprocess(augmented_crop_image)
        text_tokens = self.tokenizer([text_description])

        return {
            "target_pixel_values": query_image_tensor,
            "search_pixel_values": search_image_tensor,
            "input_ids": text_tokens[0],
            "attention_mask": (text_tokens[0] != 0).long(),
            "index": index,
        }


class Encoder(nn.Module):
    def __init__(self, model_name, proj_dim=768):
        super().__init__()

        # Load EVA-CLIP model
        try:
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name, cache_dir=CACHE_DIR
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.vision_model = self.model.visual

            # Get feature dimensions
            self.feature_dim = 768
            self.text_feature_dim = 512
        except Exception as e:
            print(f"Error loading EVA-CLIP model: {e}")
            raise

        # Vision Pooling
        self.pool = nn.AdaptiveAvgPool2d((3, 3))

        # Projection Heads
        self.vision_projector_query = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, proj_dim),
        )

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.text_feature_dim * 2, self.feature_dim),
        )

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim * 2),
            nn.ReLU(),
            nn.Linear(proj_dim * 2, proj_dim),
        )

    def ref_forward(self, pixel_values):
        # 1. Forward through the patch embedding and initial layers
        x = self.vision_model.trunk.forward_features(pixel_values)
        patch_tokens = x[:, 1:, :]

        # 3. Spatial Processing (Reshape to grid)
        B, N_minus_1, D = patch_tokens.shape
        H = W = int(N_minus_1**0.5)

        # Reshape and pool as per your existing logic
        patch_tokens_spatial = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
        pooled_features = self.pool(patch_tokens_spatial)

        pooled_features = pooled_features.flatten(2).permute(0, 2, 1)
        # proj_feature = self.vision_projector_ref(pooled_features)
        return pooled_features  # [B, 9, 768]

    # def ref_forward(self, pixel_values):
    #     # EVA-CLIP vision model returns features directly
    #     features = self.vision_model(pixel_values)
    #     # Expand to [B, 9, D] by replicating
    #     B = features.shape[0]
    #     grid_features = features.unsqueeze(1).expand(B, 9, -1)

    #     projected_features = self.vision_projector(grid_features)
    #     return projected_features  # [B, 9, 768]

    def query_forward(self, pixel_values):
        x = self.vision_model.trunk.forward_features(pixel_values)
        pooled_features = x[:, 0, :]
        # proj_feature = self.vision_projector_query(pooled_features)
        return pooled_features  # [B, 768]

    def text_forward(self, input_ids, attention_mask):
        text_features = self.model.encode_text(input_ids)
        output_feature = self.text_projector(text_features)
        return output_feature  # [B, 768]

    # def forward_fusion(self, vision_feat, text_feat):
    #     combined_input = torch.cat([vision_feat, text_feat], dim=1)
    #     fused_feat = self.fusion_mlp(combined_input)
    #     return fused_feat


def info_nce_loss(query_feats, candidate_feats, positive_indices, temperature=0.07):
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature
    return F.cross_entropy(sim_matrix, positive_indices)


def main(save_path):
    exp_name = save_path.split("/")[-1] if save_path else "default_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Loading models and processor...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, cache_dir=CACHE_DIR
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    processor_wrapper = OpenClipImageProcessorWrapper(preprocess)
    tokenizer_wrapper = OpenClipTokenizerWrapper(tokenizer)

    encoder = Encoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE)
    use_cuda = DEVICE.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    print("Setting up dataset and dataloader...")
    train_dataset = ShiftedSatelliteDroneDataset(
        processor=processor_wrapper,
        processor_sat=processor_wrapper,
        tokenizer=tokenizer_wrapper,
        split="train",
        enable_timing_log=ENABLE_DATASET_TIMING_LOG,
        timing_log_interval=DATASET_TIMING_LOG_INTERVAL,
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

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")
    min_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        encoder.train()
        total_loss = 0
        data_wait_acc = 0.0
        compute_acc = 0.0
        iter_end_time = time.perf_counter()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for i, batch in enumerate(progress_bar):
            step_idx = i + 1

            if ENABLE_SEARCH_PIXEL_VIS and step_idx % SEARCH_PIXEL_VIS_INTERVAL == 0:
                vis_path = (
                    f"runs/visualizations/{exp_name}/"
                    f"search_pixels_e{epoch + 1}_s{step_idx}.jpg"
                )
                visualize_search_pixel_values(
                    batch["search_pixel_values"],
                    save_path=vis_path,
                    max_images=SEARCH_PIXEL_VIS_MAX_IMAGES,
                )

            iter_start_time = time.perf_counter()
            data_wait_time = iter_start_time - iter_end_time

            target_pixel_values = batch["target_pixel_values"].to(
                DEVICE, non_blocking=True
            )
            search_pixel_values = batch["search_pixel_values"].to(
                DEVICE, non_blocking=True
            )
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = (input_ids != 0).long()
            local_indices = batch["index"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=False):
                anchor_feats = encoder.query_forward(target_pixel_values)
                grid_feats = encoder.ref_forward(search_pixel_values)
                candidate_feats = grid_feats.reshape(-1, PROJECTION_DIM)
                text_feats = encoder.text_forward(input_ids, attention_mask)

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

                img_loss = info_nce_loss(anchor_feats, candidate_feats, positive_indices)
                text_loss = info_nce_loss(text_feats, candidate_feats, positive_indices)
                loss = img_loss + text_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iter_end_time = time.perf_counter()
            compute_time = iter_end_time - iter_start_time

            if ENABLE_BATCH_TIMING_LOG:
                data_wait_acc += data_wait_time
                compute_acc += compute_time
                if step_idx % BATCH_TIMING_LOG_INTERVAL == 0:
                    avg_data_wait_ms = (data_wait_acc / step_idx) * 1000.0
                    avg_compute_ms = (compute_acc / step_idx) * 1000.0
                    print(
                        "[BatchTiming] "
                        f"epoch={epoch + 1} step={step_idx} "
                        f"avg_data_wait={avg_data_wait_ms:.2f}ms "
                        f"avg_compute={avg_compute_ms:.2f}ms"
                    )
                    writer.add_scalar(
                        "Timing/train_avg_data_wait_ms",
                        avg_data_wait_ms,
                        epoch * len(train_dataloader) + i,
                    )
                    writer.add_scalar(
                        "Timing/train_avg_compute_ms",
                        avg_compute_ms,
                        epoch * len(train_dataloader) + i,
                    )

            total_loss += loss.item()
            if ENABLE_BATCH_TIMING_LOG:
                progress_bar.set_postfix(
                    {
                        "img_text_loss": f"{loss.item():.4f}",
                        "wait_ms": f"{data_wait_time * 1000.0:.1f}",
                        "comp_ms": f"{compute_time * 1000.0:.1f}",
                    }
                )
            else:
                progress_bar.set_postfix({"img_text_loss": f"{loss.item():.4f}"})
            writer.add_scalar(
                "Loss/train_batch", loss.item(), epoch * len(train_dataloader) + i
            )

        avg_loss = total_loss / len(train_dataloader)
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        print(f"Epoch {epoch + 1} finished. Average train Loss: {avg_loss:.4f}")

        if avg_loss < min_loss:
            os.makedirs(save_path, exist_ok=True)
            torch.save(encoder.state_dict(), f"{save_path}/best.pth")
            min_loss = avg_loss

    os.makedirs(save_path, exist_ok=True)
    torch.save(encoder.state_dict(), f"{save_path}/last.pth")

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
        attention_mask = (input_ids != 0).long()
        local_indices = batch["index"].to(DEVICE, non_blocking=True)

        batch_offsets = torch.arange(local_indices.shape[0], device=DEVICE) * 9
        positive_indices = local_indices + batch_offsets

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            anchor_feats = model.query_forward(target_pixel_values)
            grid_feats = model.ref_forward(search_pixel_values)
            text_feats = model.text_forward(input_ids, attention_mask)

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
    query_feat_path="eval_evaclip_query_feats.pt",
    gallery_feat_path="eval_evaclip_gallery_feats.pt",
):
    print("Stage 1/2: extracting and saving query/gallery features...")
    use_cuda = DEVICE.startswith("cuda")
    _, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, cache_dir=CACHE_DIR
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    processor_wrapper = OpenClipImageProcessorWrapper(preprocess)
    tokenizer_wrapper = OpenClipTokenizerWrapper(tokenizer)

    test_dataset = ShiftedSatelliteDroneDataset(
        processor=processor_wrapper,
        processor_sat=processor_wrapper,
        tokenizer=tokenizer_wrapper,
        split="test",
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
    model_path = checkpoint_path or "../ckpt/retrieval_evaclip/best.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    encoder.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
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
            attention_mask = (input_ids != 0).long()
            satellite_paths = batch["satellite_path"]

            with torch.amp.autocast("cuda", enabled=False):
                anchor_feats = encoder.query_forward(query_inputs)
                text_feats = encoder.text_forward(input_ids, attention_mask)
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
    query_feat_path="eval_evaclip_query_feats.pt",
    gallery_feat_path="eval_evaclip_gallery_feats.pt",
    subset_height=None,
    subset_angle=None,
    test_num=100,
):
    print("Stage 2/2: loading saved features and scoring...")
    if not os.path.exists(query_feat_path):
        raise FileNotFoundError(f"Query feature file not found: {query_feat_path}")
    if not os.path.exists(gallery_feat_path):
        raise FileNotFoundError(
            f"Gallery feature file not found: {gallery_feat_path}"
        )

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

    gallery_feats = gallery_payload["features"]
    gallery_labels = gallery_payload["labels"]

    selected_indices = []
    for idx, (h, a) in enumerate(zip(query_heights, query_angles)):
        if subset_height is not None and int(h) != int(subset_height):
            continue
        if subset_angle is not None and int(a) != int(subset_angle):
            continue
        selected_indices.append(idx)

    if not selected_indices:
        print(
            f"No queries found for filters: height={subset_height}, angle={subset_angle}."
        )
        return

    query_feats = query_feats[selected_indices]
    query_labels = [query_labels[idx] for idx in selected_indices]
    print(
        f"Scoring {len(selected_indices)} / {len(query_payload['labels'])} queries "
        f"(height={subset_height}, angle={subset_angle}, test_num={test_num})"
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
    subset_height=None,
    subset_angle=None,
    test_num=100,
    query_feat_path="eval_evaclip_query_feats.pt",
    gallery_feat_path="eval_evaclip_gallery_feats.pt",
    run_extract=True,
):
    if run_extract:
        extract_eval_features(
            checkpoint_path=checkpoint_path,
            query_feat_path=query_feat_path,
            gallery_feat_path=gallery_feat_path,
        )
    score_eval_from_saved(
        query_feat_path=query_feat_path,
        gallery_feat_path=gallery_feat_path,
        subset_height=subset_height,
        subset_angle=subset_angle,
        test_num=test_num,
    )


if __name__ == "__main__":
    exp_name = "retrieval_evaclip"
    save_dir = f"../ckpt/{exp_name}"

    if os.path.exists(save_dir):
        print(f"Experiment directory '{save_dir}' already exists.")
        print("Re-using directory. Logs and checkpoints might be overwritten.")
    else:
        os.makedirs(save_dir)
        print(f"Created experiment directory: {save_dir}")

    # Run training
    main(save_dir)
