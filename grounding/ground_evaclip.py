import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open_clip
from PIL import Image
from tqdm import tqdm
import os
from glob import glob
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import json
from einops import rearrange

from bbox.yolo_utils import (
    get_tensor_anchors,
    build_target,
    yolo_loss,
    eval_iou_acc,
    SpatialTransformer,
)

SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# --- Configuration ---
MODEL_NAME = "hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k"
CACHE_DIR = "/data/feihong/hf_cache"
DRONE_VIEW_FOLDER = "/data/feihong/drone_view"
IMAGE_FOLDER = "/data/feihong/image_1024"
BBOX_FILE = "/data/feihong/univerisity_dev/runs/bbox_isaac.json"
TRAIN_BBOX_FILE = "/data/feihong/univerisity_dev/runs/train.json"
TEST_BBOX_FILE = "/data/feihong/univerisity_dev/runs/test.json"
TEXT_FILE = "/data/feihong/drone_text_single_long.json"

SAT_ORIG_SIZE = (3840, 2160)
UNIV_SAT_SIZE = (640, 640)
DRONE_SIZE = (256, 256)

NUM_EPOCHS = 40
BATCH_SIZE = 20
LEARNING_RATE = 1e-4
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
PROJECTION_DIM = 768


def format_satellite_img_bbox(
    image,
    bbox,
    mode="train",
    target_size=UNIV_SAT_SIZE,
):
    """Crop satellite image around bbox and resize."""
    x1, y1, x2, y2 = bbox
    width, height = image.size

    min_crop = max(y2 - y1, x2 - x1) * 2
    crop_size = random.uniform(min_crop, max(min_crop, height))

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
    ratio = image.size[0] / target_size[0]
    image = image.resize(target_size)

    new_x1 = (x1 - left) / ratio
    new_y1 = (y1 - top) / ratio
    new_x2 = (x2 - left) / ratio
    new_y2 = (y2 - top) / ratio
    return image, [new_x1, new_y1, new_x2, new_y2]


def resize_drone_image(image, target_size=DRONE_SIZE):
    return image.resize(target_size, Image.Resampling.BILINEAR)


class TargetSearchDataset(Dataset):
    def __init__(
        self,
        image_pairs,
        preprocess,
        tokenizer,
        img_to_text_dict,
        bbox_dict=None,
        mode="train",
    ):
        self.image_paths = image_pairs
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.img_to_text_dict = img_to_text_dict
        self.bbox_dict = bbox_dict

        self.crop_size = 2160 // 5 * 3
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

        name = query_path.split("/")[-2]

        text_name = name + "_" + query_path.split("/")[-1].split(".")[0][-2:]
        text_description = self.img_to_text_dict.get(text_name, "")

        for noun in self.remove:
            text_description = text_description.replace(noun, "")

        try:
            query_image = Image.open(query_path).convert("RGB")
            search_image = Image.open(search_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping this item.")
            return self.__getitem__((idx + 1) % len(self))

        query_image = resize_drone_image(query_image, DRONE_SIZE)

        bbox_key = name
        bbox = None
        if self.bbox_dict:
            bbox = self.bbox_dict.get(bbox_key, None)
        if bbox is None:
            bbox = (1536, 656, 2268, 1374)

        augmented_crop_image, normalized_bbox = format_satellite_img_bbox(
            search_image, bbox, mode=self.mode, target_size=UNIV_SAT_SIZE
        )

        center_x = (normalized_bbox[0] + normalized_bbox[2]) / 2
        center_y = (normalized_bbox[1] + normalized_bbox[3]) / 2

        col_idx = min(int(center_x * 3), 2)
        row_idx = min(int(center_y * 3), 2)
        index = row_idx * 3 + col_idx

        query_image_tensor = self.preprocess(query_image)
        search_image_tensor = self.preprocess(augmented_crop_image)
        text_tokens = self.tokenizer([text_description])

        return {
            "target_pixel_values": query_image_tensor,
            "search_pixel_values": search_image_tensor,
            "input_ids": text_tokens[0],
            "attention_mask": (text_tokens[0] != 0).long(),
            "index": index,
            "bbox": torch.tensor(normalized_bbox, dtype=torch.float32),
        }


class Encoder(nn.Module):
    def __init__(self, model_name=MODEL_NAME, proj_dim=768):
        super().__init__()

        try:
            self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
                model_name, cache_dir=CACHE_DIR
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.vision_model = self.model.visual

            self.feature_dim = 768
            self.text_feature_dim = 512
        except Exception as e:
            print(f"Error loading EVA-CLIP model: {e}")
            raise

        self.pool_sat = nn.AdaptiveAvgPool2d((20, 20))
        self.pool_dro = nn.AdaptiveAvgPool2d((8, 8))
        self.position_embedding = nn.Embedding(400, PROJECTION_DIM)
        self.register_buffer(
            "position_ids", torch.arange(400).expand((1, -1)), persistent=False
        )

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.text_feature_dim * 2, proj_dim),
        )

        self.bbox_transformer = SpatialTransformer(
            in_channels=self.feature_dim,
            n_heads=8,
            d_head=64,
            depth=1,
            context_dim=self.feature_dim,
        )

        self.bbox_fcn_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.feature_dim,
                out_channels=self.feature_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim // 2, 9 * 5, kernel_size=1),
        )

        self.bbox_coords_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.feature_dim,
                out_channels=self.feature_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim // 2, 1, kernel_size=1),
        )

    def sat_forward(self, pixel_values):
        """Extract satellite image features from last hidden state."""
        x = self.vision_model.trunk.forward_features(pixel_values)
        patch_tokens = x[:, 1:, :]

        B, N, D = patch_tokens.shape
        H = W = int(N**0.5)

        sat_feat = self.pool_sat(patch_tokens.permute(0, 2, 1).reshape(B, D, H, W))
        return sat_feat

    def query_forward(self, pixel_values):
        x = self.vision_model.trunk.forward_features(pixel_values)
        pooled_features = x[:, 0, :]
        return pooled_features

    def text_forward(self, input_ids, attention_mask=None):
        text_features = self.model.encode_text(input_ids)
        proj_feature = self.text_projector(text_features)
        return proj_feature

    def bbox_forward(self, anchor_pixel_values, search_pixel_values):
        """
        Combined forward for bbox prediction + retrieval.

        Returns:
            pred_anchor: YOLO predictions (B, 45, H, W)
            pred_coords: (B, 1, H, W)
            anchor_feats: Query features [B, D]
            anchor_feats_pooled: Pooled query features [B, D, 8, 8]
        """
        B = anchor_pixel_values.shape[0]

        anchor_feats_full = self.vision_model.trunk.forward_features(
            anchor_pixel_values
        )[:, 1:, :]
        anchor_feats_pooled = self._pool_grid_features(anchor_feats_full)
        anchor_feats = anchor_feats_full[:, 0, :]

        grid_features = self.sat_forward(search_pixel_values)

        anchor_context = rearrange(
            anchor_feats_pooled, "b c h w -> b (h w) c"
        ).contiguous()

        fused_features = self.bbox_transformer(x=grid_features, context=anchor_context)

        pred_anchor = self.bbox_fcn_out(fused_features)
        pred_coords = self.bbox_coords_out(fused_features)

        return pred_anchor, pred_coords, anchor_feats, anchor_feats_pooled

    def _pool_grid_features(self, vision_features):
        B, N, D = vision_features.shape
        H = W = int(N**0.5)

        patch_tokens = vision_features.permute(0, 2, 1).reshape(B, D, H, W)
        pooled_features = self.pool_dro(patch_tokens)

        return pooled_features


def info_nce_loss(query_feats, candidate_feats, positive_indices, temperature=0.07):
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature
    return F.cross_entropy(sim_matrix, positive_indices)


def validation(loader, model, anchors_full, img_size):
    """Validate model."""
    model.eval()

    accu50_meter = AverageMeter()
    accu25_meter = AverageMeter()
    iou_meter = AverageMeter()

    for batch in tqdm(loader, desc="Validating"):
        query_imgs = batch["target_pixel_values"].to(DEVICE)
        rs_imgs = batch["search_pixel_values"].to(DEVICE)
        ori_gt_bbox = batch["bbox"].to(DEVICE)

        with torch.no_grad():
            pred_anchor, _, _, _ = model.bbox_forward(query_imgs, rs_imgs)

        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        _, best_anchor_gi_gj = build_target(
            ori_gt_bbox, anchors_full, img_size, pred_anchor.shape[3]
        )

        accu_list, accu_center, iou, _, _, _ = eval_iou_acc(
            pred_anchor,
            ori_gt_bbox,
            anchors_full,
            best_anchor_gi_gj[:, 1],
            best_anchor_gi_gj[:, 2],
            img_size,
            iou_threshold_list=[0.5, 0.25],
        )

        accu50_meter.update(accu_list[0].item(), query_imgs.shape[0])
        accu25_meter.update(accu_list[1].item(), query_imgs.shape[0])
        iou_meter.update(iou.item(), query_imgs.shape[0])

    return accu50_meter.avg, accu25_meter.avg, iou_meter.avg


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
    model, preprocess, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, cache_dir=CACHE_DIR
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    encoder = Encoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
    encoder.train()

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE)

    img_to_text_dict = json.load(open(TEXT_FILE, "r"))
    train_bbox_dict = json.load(open(TRAIN_BBOX_FILE, "r"))
    test_bbox_dict = json.load(open(TEST_BBOX_FILE, "r"))

    anchors_full = get_tensor_anchors(DEVICE)

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

    train_dataset = TargetSearchDataset(
        image_pairs=train_image_pairs,
        preprocess=preprocess,
        tokenizer=tokenizer,
        img_to_text_dict=img_to_text_dict,
        bbox_dict=train_bbox_dict,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4
    )

    test_dataset = TargetSearchDataset(
        image_pairs=test_image_pairs,
        preprocess=preprocess,
        tokenizer=tokenizer,
        img_to_text_dict=img_to_text_dict,
        bbox_dict=test_bbox_dict,
        mode="test",
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4
    )

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")
    max_iou = 0

    for epoch in range(NUM_EPOCHS):
        encoder.train()
        total_loss = 0
        total_bbox_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for i, batch in enumerate(progress_bar):
            target_pixel_values = batch["target_pixel_values"].to(DEVICE)
            search_pixel_values = batch["search_pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            local_indices = batch["index"].to(DEVICE)
            target_bbox = batch["bbox"].to(DEVICE)

            pred_anchor, pred_coords, anchor_feats, grid_feats = encoder.bbox_forward(
                target_pixel_values, search_pixel_values
            )

            B = pred_anchor.shape[0]
            pred_anchor = pred_anchor.view(
                B, 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
            )

            new_gt_bbox, best_anchor_gi_gj = build_target(
                target_bbox,
                anchors_full,
                UNIV_SAT_SIZE[0],
                pred_anchor.shape[3],
            )

            loss_geo, loss_cls = yolo_loss(
                pred_anchor,
                new_gt_bbox,
                anchors_full,
                best_anchor_gi_gj,
                UNIV_SAT_SIZE[0],
            )
            bbox_loss = loss_geo + loss_cls

            loss = bbox_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "bbox_loss": f"{bbox_loss.item():.4f}",
                }
            )
            writer.add_scalar(
                "Loss/train_batch", loss.item(), epoch * len(train_dataloader) + i
            )

        avg_loss = total_loss / len(train_dataloader)
        avg_bbox_loss = total_bbox_loss / len(train_dataloader)

        encoder.eval()
        accu50, accu25, val_iou = validation(
            test_dataloader, encoder, anchors_full, UNIV_SAT_SIZE[0]
        )
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Metrics/val_iou", val_iou, epoch)
        writer.add_scalar("Metrics/val_accu50", accu50, epoch)
        writer.add_scalar("Metrics/val_accu25", accu25, epoch)
        print(
            f"Epoch {epoch + 1} finished. "
            f"Avg train Loss: {avg_loss:.4f} (bbox: {avg_bbox_loss:.4f}). "
            f"IoU: {val_iou:.4f}, Accu50: {accu50:.4f}, Accu25: {accu25:.4f}"
        )

        if val_iou > max_iou:
            os.makedirs(save_path, exist_ok=True)
            torch.save(encoder.state_dict(), f"{save_path}/best.pth")
            max_iou = val_iou
            not_update = 0
        else:
            not_update += 1
        if not_update > 5:
            print(f"Validation loss not improving {not_update} epoch.")

    print("Training complete.")
    writer.close()


if __name__ == "__main__":
    exp_name = "ground_evaclip"
    save_dir = f"../ckpt/{exp_name}"

    if os.path.exists(save_dir):
        print(f"Experiment directory '{save_dir}' already exists.")
        print("Re-using directory. Logs and checkpoints might be overwritten.")
    else:
        os.makedirs(save_dir)
        print(f"Created experiment directory: {save_dir}")

    main(save_dir)
