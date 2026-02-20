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
from einops import rearrange, einsum
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from model import Encoder_gem as Encoder_attn
from bbox.yolo_utils import (
    get_tensor_anchors,
    build_target,
    yolo_loss,
    eval_iou_acc,
    SpatialTransformer,
)

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
MODEL_NAME = "google/siglip-base-patch16-224"
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

NUM_EPOCHS = 200
BATCH_SIZE = 15
GRAD_ACCUMULATION_STEPS = 3
LEARNING_RATE = 1e-5
LR_MIN = 1e-10
COSINE_EPOCHS = 200
BBOX_LOSS_WEIGHT = 0.1
PROJECTION_DIM = 768


def get_loss_weights(epoch, num_epochs):
    """Cosine annealing for loss weights.

    Returns:
        bbox_weight: decreases from 0.99 to 0.01
        retrieval_weight: increases from 0.01 to 0.99
    """
    progress = epoch / num_epochs
    bbox_weight = 0.5 * (1 + np.cos(2 * np.pi * progress)) / 2
    retrieval_weight = 1.0 - bbox_weight
    return bbox_weight, retrieval_weight


def format_satellite_img_bbox(
    image,
    bbox,
    mode="train",
    target_size=UNIV_SAT_SIZE,
):
    """Crop satellite image around bbox and resize."""
    x1, y1, x2, y2 = bbox
    width, height = image.size

    min_crop = max(y2 - y1, x2 - x1) * 1.2
    crop_size = random.uniform(min_crop, max(min_crop, height))
    # crop_size = 2160 // 5 * 3
    if mode == "test":
        crop_size = height

    min_left = 0
    max_left = width - crop_size
    min_top = 0
    max_top = height - crop_size

    # Now apply the bbox constraints
    # These will tighten the range to ensure the bbox is inside the crop
    min_left = max(min_left, x2 - crop_size)
    min_top = max(min_top, y2 - crop_size)
    max_left = min(max_left, x1)
    max_top = min(max_top, y1)

    if max_left < min_left:
        max_left = min_left
    if max_top < min_top:
        max_top = min_top

    left = random.uniform(min_left, max_left)
    top = random.uniform(min_top, max_top)
    if mode == "test":
        left = 840
        top = 0

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
            for number in [
                "01.",
                "21.",
                "31.",
                "41.",
                "51.",
            ]:
                new_query_path = query_path.replace("01.", number)
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
        # normalized_bbox = [i / UNIV_SAT_SIZE[0] for i in normalized_bbox]
        center_x = (normalized_bbox[0] + normalized_bbox[2]) / 2 / UNIV_SAT_SIZE[0]
        center_y = (normalized_bbox[1] + normalized_bbox[3]) / 2 / UNIV_SAT_SIZE[0]

        col_idx = min(int(center_x * 3), 2)
        row_idx = min(int(center_y * 3), 2)
        index = row_idx * 3 + col_idx

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
    def __init__(self, model_name=MODEL_NAME, proj_dim=768):
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
        self.pool_sat = nn.AdaptiveAvgPool2d((20, 20))
        self.pool_dro = nn.AdaptiveAvgPool2d((8, 8))
        self.pool_info = nn.AdaptiveAvgPool2d((3, 3))
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
            bbox_pred: YOLO predictions (B, 45, H, W)
            anchor_feats: B×768 (query features for InfoNCE)
            grid_feats: B×N×768 (pooled grid features for InfoNCE)
        """
        B = anchor_pixel_values.shape[0]

        anchor_feats = self.vision_model(anchor_pixel_values).last_hidden_state

        embedding = self.vision_model.embeddings.patch_embedding(search_pixel_values)
        embedding_sat = self.pool_sat(embedding).flatten(2).transpose(
            1, 2
        ) + self.position_embedding(self.position_ids)
        grid_features = self.vision_model.encoder(embedding_sat).last_hidden_state

        anchor_feats_pooled = self._pool_grid_features(anchor_feats)

        N = grid_features.shape[1]
        H = W = int(N**0.5)
        grid_features_2d = grid_features.permute(0, 2, 1).reshape(
            B, self.feature_dim, H, W
        )

        grid_features_info = (
            self.pool_info(grid_features_2d)
            .reshape(B, self.feature_dim, 9)
            .permute(0, 2, 1)
        )
        anchor_context = rearrange(
            anchor_feats_pooled, "b c h w -> b (h w) c"
        ).contiguous()

        fused_features = self.bbox_transformer(
            x=grid_features_2d, context=anchor_context
        )

        pred_anchor = self.bbox_fcn_out(fused_features)
        pred_coords = self.bbox_coords_out(fused_features)

        return (
            pred_anchor,
            pred_coords,
            anchor_feats.mean(1),
            grid_features_info,
        )

    def _pool_grid_features(self, vision_features):
        B, N, D = vision_features.shape
        H = W = int(N**0.5)

        patch_tokens = vision_features.permute(0, 2, 1).reshape(B, D, H, W)
        pooled_features = self.pool_dro(patch_tokens)

        return pooled_features

    def preprocess(self, query_image, search_image):
        if not isinstance(getattr(self, "processor", None)):
            self.processor = AutoImageProcessor.from_pretrained(
                MODEL_NAME, cache_dir=CACHE_DIR
            )
            self.processor_sat = AutoImageProcessor.from_pretrained(
                MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
            )
        query_inputs = self.processor(images=query_image, return_tensors="pt")
        search_inputs = self.processor_sat(images=search_image, return_tensors="pt")
        return query_inputs["pixel_values"][0], search_inputs["pixel_values"][0]





def info_nce_loss(query_feats, candidate_feats, positive_indices, temperature=0.07):
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature
    return F.cross_entropy(sim_matrix, positive_indices)


def validation(loader, model, accelerator, anchors_full, img_size):
    """Validate model."""
    model.eval()

    accu50_meter = AverageMeter()
    accu25_meter = AverageMeter()
    iou_meter = AverageMeter()
    info_nce_meter = AverageMeter()

    for batch in tqdm(loader, desc="Validating"):
        query_imgs = batch["target_pixel_values"]
        rs_imgs = batch["search_pixel_values"]
        ori_gt_bbox = batch["bbox"]
        input_ids = batch["input_ids"]
        local_indices = batch["index"]

        with torch.no_grad():
            pred_anchor, pred_coords, anchor_feats, grid_feats, fused_feats = model(
                query_imgs, rs_imgs, input_ids
            )

        B = pred_anchor.shape[0]
        pred_anchor = pred_anchor.view(
            B, 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        new_gt_bbox, best_anchor_gi_gj = build_target(
            ori_gt_bbox,
            anchors_full,
            UNIV_SAT_SIZE[0],
            pred_anchor.shape[3],
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

        candidate_feats = grid_feats.reshape(-1, PROJECTION_DIM)

        positive_indices = torch.zeros(
            local_indices.shape[0],
            local_indices.shape[0] * 9,
            device=accelerator.device,
        )
        batch_offsets = (
            torch.arange(local_indices.shape[0], device=accelerator.device) * 9
        )
        row_indices_broad = torch.arange(
            local_indices.shape[0], device=accelerator.device
        ).unsqueeze(1)
        col_offsets = torch.arange(9, device=accelerator.device).unsqueeze(0)
        same_image_cols = batch_offsets.unsqueeze(1) + col_offsets

        positive_indices[row_indices_broad, same_image_cols] = 0.5
        global_positive_indices = local_indices + batch_offsets
        row_indices_flat = torch.arange(
            local_indices.shape[0], device=accelerator.device
        )
        positive_indices[row_indices_flat, global_positive_indices] = 0.95

        img_text_loss = info_nce_loss(
            fused_feats, candidate_feats, positive_indices
        )
        # img_text_loss += 0.7 * info_nce_loss(
        #     anchor_feats, candidate_feats, positive_indices
        # )
        accu50_meter.update(accu_list[0].item(), query_imgs.shape[0])
        accu25_meter.update(accu_list[1].item(), query_imgs.shape[0])
        iou_meter.update(iou.item(), query_imgs.shape[0])
        info_nce_meter.update(img_text_loss.item(), query_imgs.shape[0])

    return accu50_meter.avg, accu25_meter.avg, iou_meter.avg, info_nce_meter.avg


def main(save_path):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        kwargs_handlers=[ddp_kwargs],
    )

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
    accelerator.print(f"Found {len(image_pairs)} valid pairs.")

    print("Loading models and processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor_sat = AutoImageProcessor.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
    )

    model = Encoder_attn(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM)
    # checkpoint_path = "/data/feihong/ckpt/unified_siglip_base/retrieval.pth"
    # new_dict = {
    #     k.replace("_orig_mod.", ""): v
    #     for k, v in torch.load(checkpoint_path, map_location="cpu").items()
    # }
    # model.load_state_dict(new_dict)

    img_to_text_dict = json.load(open(TEXT_FILE, "r"))
    train_bbox_dict = json.load(open(TRAIN_BBOX_FILE, "r"))
    test_bbox_dict = json.load(open(TEST_BBOX_FILE, "r"))

    anchors_full = get_tensor_anchors(accelerator.device)

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
    train_dataset = TargetSearchDataset(
        image_pairs=train_image_pairs,
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        img_to_text_dict=img_to_text_dict,
        bbox_dict=train_bbox_dict,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
    )

    test_dataset = TargetSearchDataset(
        image_pairs=test_image_pairs,
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        img_to_text_dict=img_to_text_dict,
        bbox_dict=test_bbox_dict,
        mode="test",
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_EPOCHS, eta_min=LR_MIN
    )

    model, optimizer, scheduler, train_dataloader, test_dataloader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, test_dataloader
        )
    )

    accelerator.print(f"Starting training for {NUM_EPOCHS} epochs...")
    max_iou = 0
    min_info = 1000

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_bbox_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for i, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                target_pixel_values = batch["target_pixel_values"]
                search_pixel_values = batch["search_pixel_values"]
                input_ids = batch["input_ids"]
                local_indices = batch["index"]
                target_bbox = batch["bbox"]

                pred_anchor, pred_coords, anchor_feats, grid_feats, fused_feats = model(
                    target_pixel_values, search_pixel_values, input_ids
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

                candidate_feats = grid_feats.reshape(-1, PROJECTION_DIM)
                positive_indices = torch.zeros(B, B * 9, device=accelerator.device)

                # Create the indices for the local batch
                batch_offsets = torch.arange(B, device=accelerator.device) * 9
                row_indices_broad = torch.arange(
                    B, device=accelerator.device
                ).unsqueeze(1)
                col_offsets = torch.arange(9, device=accelerator.device).unsqueeze(0)

                # Logic: Neighbors get 0.5 (or 0.0), Exact match gets 0.95 (or 1.0)
                same_image_cols = batch_offsets.unsqueeze(1) + col_offsets
                positive_indices[row_indices_broad, same_image_cols] = (
                    0.5  # or 0.1 for sharpness
                )

                global_positive_indices = local_indices + batch_offsets
                row_indices_flat = torch.arange(B, device=accelerator.device)
                positive_indices[row_indices_flat, global_positive_indices] = 0.95
                img_text_loss = 0.6 * info_nce_loss(
                    fused_feats, candidate_feats, positive_indices
                )
                img_text_loss += 0.4 * info_nce_loss(
                    anchor_feats, candidate_feats, positive_indices
                )

                bbox_weight, retrieval_weight = get_loss_weights(epoch, NUM_EPOCHS)
                loss = retrieval_weight * img_text_loss + bbox_weight * bbox_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "bbox_loss": f"{bbox_loss.item():.4f}",
                    "img_text_loss": f"{img_text_loss.item():.4f}",
                }
            )
            writer.add_scalar(
                "Loss/train_batch", loss.item(), epoch * len(train_dataloader) + i
            )
            writer.add_scalar(
                "Weights/bbox", bbox_weight, epoch * len(train_dataloader) + i
            )
            writer.add_scalar(
                "Weights/retrieval", retrieval_weight, epoch * len(train_dataloader) + i
            )

        avg_loss = total_loss / len(train_dataloader)
        avg_bbox_loss = total_bbox_loss / len(train_dataloader)

        model.eval()
        accu50, accu25, val_iou, val_info_nce = validation(
            test_dataloader,
            model,
            accelerator,
            anchors_full,
            UNIV_SAT_SIZE[0],
        )
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Metrics/val_iou", val_iou, epoch)
        writer.add_scalar("Metrics/val_accu50", accu50, epoch)
        writer.add_scalar("Metrics/val_accu25", accu25, epoch)
        accelerator.print(
            f"Epoch {epoch + 1} finished. "
            f"Avg train Loss: {avg_loss:.4f} (bbox: {avg_bbox_loss:.4f}). "
            f"IoU: {val_iou:.4f}, Accu50: {accu50:.4f}, Accu25: {accu25:.4f}, InfoNCE: {val_info_nce:.4f}"
        )

        if accelerator.is_main_process:
            if val_info_nce < min_info:
                os.makedirs(save_path, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)

                save_filename = f"{save_path}/best_info_{not_update}.pth"
                accelerator.save(unwrapped_model.state_dict(), save_filename)

                print(f"Saved checkpoint info: {save_filename}")
                min_info = val_info_nce
                not_update += 1
            if max_iou < val_iou:
                os.makedirs(save_path, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)

                save_filename = f"{save_path}/best_iou_{not_update}.pth"
                accelerator.save(unwrapped_model.state_dict(), save_filename)

                print(f"Saved checkpoint iou: {save_filename}")
                max_iou = val_iou
                not_update += 1
        scheduler.step()

    accelerator.print("Training complete.")
    writer.close()


def eval(run=False):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 43
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    class EvalDataset(Dataset):
        def __init__(
            self, eval_list, processor, processor_sat, tokenizer, img_to_text_dict
        ):
            self.eval_list = eval_list
            self.processor = processor
            self.processor_sat = processor_sat
            self.tokenizer = tokenizer
            self.img_to_text_dict = img_to_text_dict
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
            return len(self.eval_list)

        def __getitem__(self, idx):
            query_path = self.eval_list[idx]
            name = query_path.split("/")[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"

            # 1. Load Text
            text_name = name + "_01"
            text_description = self.img_to_text_dict.get(text_name, "")
            for noun in self.remove:
                text_description = text_description.replace(noun, "")

            # 2. Load Images
            # Note: We return None if file missing to handle it in collate_fn or loop
            if not os.path.exists(search_path):
                return None

            try:
                query_image = Image.open(query_path).convert("RGB")
                search_image = Image.open(search_path).convert("RGB")
            except Exception:
                return None

            # 3. Preprocess
            # Resize drone image
            query_image = resize_drone_image(query_image, DRONE_SIZE)

            # Format satellite image (Center crop for testing as per original logic)
            bbox = [0, 0, 0, 0]  # Dummy bbox for test mode

            search_image = search_image.crop((840, 0, 3000, 2160))
            search_image = search_image.resize(UNIV_SAT_SIZE)
            # 4. Processor Transforms
            # We return the raw tensors, not the dict, to make batching easier
            query_pixels = self.processor(images=query_image, return_tensors="pt")[
                "pixel_values"
            ][0]
            search_pixels = self.processor_sat(
                images=search_image, return_tensors="pt"
            )["pixel_values"][0]

            input_ids = self.tokenizer(
                text_description,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )["input_ids"][0]

            return {
                "name": name,
                "query_pixels": query_pixels,
                "search_pixels": search_pixels,
                "input_ids": input_ids,
            }

    # --- Main Eval Logic ---

    def calcu_cos(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b_norm, a_norm[..., None]).flatten()

    if run:
        print("Initializing Evaluation...")
        img_to_text_dict = json.load(
            open("/data/feihong/drone_text_single_long.json", "r")
        )

        # Load Processors
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        processor_sat = AutoImageProcessor.from_pretrained(
            MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Load Model
        encoder = Encoder_attn(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(
            DEVICE
        )
        # Using compile for speedup
        # encoder = torch.compile(encoder)

        model_path = f"../ckpt/unified_siglip_attn_move/best_info_58.pth"
        if os.path.exists(model_path):
            encoder.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded checkpoint: {model_path}")
        else:
            print("error")
        encoder.eval()

        # Prepare Data List
        eval_list = []
        with open("/data/feihong/ckpt/test.txt", "r") as f:
            for line in f:
                eval_list.append(line.strip())

        # Create Dataset & Loader
        dataset = EvalDataset(
            eval_list, processor, processor_sat, tokenizer, img_to_text_dict
        )

        # Custom collate to filter out None (missing files)
        def collate_fn(batch):
            batch = [b for b in batch if b is not None]
            if len(batch) == 0:
                return None
            return torch.utils.data.dataloader.default_collate(batch)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,  # Make sure BATCH_SIZE is > 1 (e.g., 32)
            shuffle=False,
            num_workers=8,  # Parallel loading
            pin_memory=True,  # Faster transfer to GPU
            collate_fn=collate_fn,
        )

        res_search = {}
        res_fused_query = {}

        print(f"Starting batched inference on {len(dataset)} items...")

        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda"),
        ):  # Mixed Precision + No Grad
            for batch in tqdm(loader):
                if batch is None:
                    continue

                names = batch["name"]
                query_inputs = batch["query_pixels"].to(DEVICE, non_blocking=True)
                search_inputs = batch["search_pixels"].to(DEVICE, non_blocking=True)
                input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)

                # Forward Pass - unpack 5 values including text_feats
                _, _, anchor_feats, grid_feats, text_feats = encoder.forward(
                    query_inputs, search_inputs, input_ids
                )

                # Feature Fusion (Vectorized)
                # Ensure dimensions match: (B, D)
                fused_query_feats = text_feats + anchor_feats

                # Move to CPU and store
                grid_feats_np = grid_feats.float().cpu().numpy()
                fused_feats_np = fused_query_feats.float().cpu().numpy()

                for i, name in enumerate(names):
                    res_search[name] = grid_feats_np[i]
                    res_fused_query[name] = fused_feats_np[i]

        np.savez("eval_search_clip.npz", **res_search)
        np.savez("eval_fused_query_clip.npz", **res_fused_query)
        print("Evaluation feature extraction complete.")
    else:
        search_res = np.load("eval_search_clip.npz")
        fused_query_res = np.load("eval_fused_query_clip.npz")

        distances = json.load(open("/data/feihong/ckpt/distances.json", "r"))
        test_num = 100
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

            img_res = np.array(res[key]).max(1).argsort()[-15:][::-1]

            if any(cand in candidate_indices for cand in img_res[:1]):
                top1 += 1
            if any(cand in candidate_indices for cand in img_res[:5]):
                top5 += 1
            if any(cand in candidate_indices for cand in img_res[:10]):
                top10 += 1

        print(f"Top 1: {top1} / {len(test_list)} ({top1 / len(test_list) * 100:.2f}%)")
        print(f"Top 5: {top5} / {len(test_list)} ({top5 / len(test_list) * 100:.2f}%)")
        print(
            f"Top 10: {top10} / {len(test_list)} ({top10 / len(test_list) * 100:.2f}%)"
        )


if __name__ == "__main__":
    exp_name = "unified_siglip_base"
    exp_name = "unified_siglip_attn"
    exp_name = "unified_siglip_attn_move"
    save_dir = f"../ckpt/{exp_name}"

    if os.path.exists(save_dir):
        print(f"Experiment directory '{save_dir}' already exists.")
        print("Re-using directory. Logs and checkpoints might be overwritten.")
    else:
        os.makedirs(save_dir)
        print(f"Created experiment directory: {save_dir}")

    # Run training
    # main(save_dir)

    # Run evaluation
    eval(True)  # Extract features
    eval(False)  # Calculate metrics

    # eval_denseuav(True)  # Extract features
    # eval_denseuav(False)  # Extract features
