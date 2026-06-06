# --- Configuration ---
import os
import json
import random
import time
from contextlib import nullcontext
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

import numpy as np
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from clearml import Task, Logger

from model import Encoder_dino
from bbox.yolo_utils import (
    get_tensor_anchors,
    build_target,
    yolo_loss,
    eval_iou_acc,
)
from dataset import ShiftedSatelliteDroneDataset

cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --- Configuration ---
class Config:
    CLEARML_ENABLED = True
    CLEARML_PROJECT = "unified_siglip"
    CLEARML_TASK_NAME = None
    MODEL_NAME = "google/siglip-base-patch16-224"
    CACHE_DIR = "/data/feihong/hf_cache"
    DRONE_VIEW_FOLDER = "/data/feihong/drone_view"
    IMAGE_FOLDER = "/data/feihong/image_1024"
    HEADING_FOLDER = "/data/feihong/range_250"
    TEXT_FILE = "/data/feihong/ckpt/drone_text_single_long.json"
    TRAIN_BBOX_FILE = "/data/feihong/univerisity_dev/runs/train.json"
    TEST_BBOX_FILE = "/data/feihong/univerisity_dev/runs/test.json"
    SATELLITE_FOLDER = "/data/feihong/asian_univ"

    SAT_ORIG_SIZE = (3840, 2160)
    UNIV_SAT_SIZE = {"height": 432, "width": 768}
    # UNIV_SAT_SIZE = (640, 640)
    DRONE_SIZE = (256, 256)

    NUM_EPOCHS = 16
    BATCH_SIZE = 8
    GRAD_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 5e-5
    LR_MIN = 1e-10
    COSINE_EPOCHS = 16
    BBOX_LOSS_WEIGHT = 0.1
    HEADING_LOSS_WEIGHT = 0.01
    PROJECTION_DIM = 1280
    FREEZE_VISION_BLOCKS = 15
    FREEZE_PATCH_GENERATOR = True
    FREEZE_CLS_TOKEN = True

    NUM_WORKERS_TRAIN = 8
    NUM_WORKERS_VAL = 4
    NUM_WORKERS_EVAL = 8
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 4
    DROP_LAST_TRAIN = True

    USE_AMP = True
    ENABLE_TF32 = True
    USE_ANGLE_INPUT = True
    USE_TEXT_INPUT = True
    OPTIMIZE_OBJECTIVE = "combined"  # combined | img_text_only | bbox_only


# --- Utility Functions ---
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


def _to_clearml_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _to_clearml_serializable(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_clearml_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_dataloader_kwargs(num_workers: int, drop_last: bool = False) -> Dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": Config.PIN_MEMORY,
        "drop_last": drop_last,
        "persistent_workers": Config.PERSISTENT_WORKERS and num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = Config.PREFETCH_FACTOR
    return kwargs


def get_loss_weights(
    epoch: int, num_epochs: int, end_num: float
) -> Tuple[float, float]:
    progress = epoch / num_epochs
    bbox_weight = (end_num - 0.5) * progress + 0.5
    retrieval_weight = 1.0 - bbox_weight
    return bbox_weight, retrieval_weight


def format_satellite_img_bbox(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    mode: str = "train",
    target_size: Tuple[int, int] = Config.UNIV_SAT_SIZE,
) -> Tuple[Image.Image, List[float]]:
    x1, y1, x2, y2 = bbox
    width, height = image.size

    min_crop = max(y2 - y1, x2 - x1) * 1.2
    crop_size = random.uniform(min_crop, max(min_crop, height))
    if mode == "test":
        crop_size = height

    min_left = max(0, x2 - crop_size)
    max_left = min(width - crop_size, x1)
    min_top = max(0, y2 - crop_size)
    max_top = min(height - crop_size, y1)

    if max_left < min_left:
        max_left = min_left
    if max_top < min_top:
        max_top = min_top

    if mode == "test":
        left, top = 840, 0
    else:
        left = random.uniform(min_left, max_left)
        top = random.uniform(min_top, max_top)

    right = left + crop_size
    bottom = top + crop_size

    image = image.crop((left, top, right, bottom))
    crop_w, crop_h = image.size
    target_w, target_h = target_size
    image = image.resize(target_size)

    # x and y require separate scales when target width != target height.
    new_x1 = (x1 - left) * (target_w / crop_w)
    new_y1 = (y1 - top) * (target_h / crop_h)
    new_x2 = (x2 - left) * (target_w / crop_w)
    new_y2 = (y2 - top) * (target_h / crop_h)

    return image, [new_x1, new_y1, new_x2, new_y2]


def resize_drone_image(
    image: Image.Image, target_size: Tuple[int, int] = Config.DRONE_SIZE
) -> Image.Image:
    return image.resize(target_size, Image.Resampling.BILINEAR)


def visualize_batch(
    batch: Dict, save_dir: str = "runs/visualizations", max_samples: int = 6, sat_size: Tuple[int, int] = Config.UNIV_SAT_SIZE
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    os.makedirs(save_dir, exist_ok=True)

    target_pixels = batch["target_pixel_values"]
    search_pixels = batch["search_pixel_values"]
    bboxes = batch.get("bbox", None)
    names = batch.get("name", [f"sample_{i}" for i in range(target_pixels.shape[0])])

    for i in range(min(target_pixels.shape[0], max_samples)):
        target_img = target_pixels[i].cpu().numpy()
        target_img = np.transpose(target_img, (1, 2, 0))
        target_img = (target_img * 0.5 + 0.5).clip(0, 1)

        search_img = search_pixels[i].cpu().numpy()
        search_img = np.transpose(search_img, (1, 2, 0))
        search_img = (search_img * 0.5 + 0.5).clip(0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Query: {names[i]}", fontsize=14, fontweight="bold")

        axes[0].imshow(target_img)
        axes[0].set_title("Target (Drone View)", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(search_img)
        axes[1].set_title("Search (Satellite View)", fontsize=12)

        if bboxes is not None:
            bbox = bboxes[i].detach().cpu().float().tolist()
            x1, y1, x2, y2 = bbox

            h, w = search_img.shape[0], search_img.shape[1]
            sx = w / float(sat_size[0])
            sy = h / float(sat_size[1])

            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy

            x1 = max(0.0, min(x1, w - 1))
            y1 = max(0.0, min(y1, h - 1))
            x2 = max(0.0, min(x2, w - 1))
            y2 = max(0.0, min(y2, h - 1))

            rect = patches.Rectangle(
                (x1, y1),
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            axes[1].add_patch(rect)

        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{names[i]}_combined.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(
        f"Saved {min(target_pixels.shape[0], max_samples)} visualizations to {save_dir}"
    )



# --- Loss Functions ---
def info_nce_loss(
    query_feats: torch.Tensor,
    candidate_feats: torch.Tensor,
    positive_indices: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature
    return F.cross_entropy(sim_matrix, positive_indices)


# --- Validation ---
def validate(
    loader: DataLoader,
    model: nn.Module,
    accelerator: Accelerator,
    anchors_full: torch.Tensor,
    img_size: Tuple[int, int],
    useap: bool = True,
) -> Tuple[float, float, float, float]:
    model.eval()

    accu50_sum = torch.tensor(0.0, device=accelerator.device)
    accu25_sum = torch.tensor(0.0, device=accelerator.device)
    iou_sum = torch.tensor(0.0, device=accelerator.device)
    info_nce_sum = torch.tensor(0.0, device=accelerator.device)
    sample_count = torch.tensor(0.0, device=accelerator.device)

    amp_enabled = Config.USE_AMP and accelerator.device.type == "cuda"

    for batch in tqdm(loader, desc="Validating", disable=not accelerator.is_main_process):
        query_imgs = batch["target_pixel_values"].to(
            accelerator.device, non_blocking=True
        )
        rs_imgs = batch["search_pixel_values"].to(
            accelerator.device, non_blocking=True
        )
        ori_gt_bbox = batch["bbox"].to(accelerator.device, non_blocking=True)
        input_ids = (
            batch["input_ids"].to(accelerator.device, non_blocking=True)
            if Config.USE_TEXT_INPUT
            else None
        )
        local_indices = batch["index"].to(accelerator.device, non_blocking=True)
        angles = (
            batch["angle"].to(accelerator.device, non_blocking=True)
            if Config.USE_ANGLE_INPUT
            else None
        )

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
            (
                pred_anchor,
                _,
                text_feats,
                anchor_feats,
                grid_feats,
                fused_feats,
            ) = model(query_imgs, rs_imgs, input_ids, angles)

        B = pred_anchor.shape[0]
        pred_anchor = pred_anchor.view(
            B, 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        _, best_anchor_gi_gj = build_target(
            ori_gt_bbox,
            anchors_full,
            (Config.UNIV_SAT_SIZE[0], Config.UNIV_SAT_SIZE[1]),
            (pred_anchor.shape[4], pred_anchor.shape[3]),
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

        candidate_feats = grid_feats.reshape(-1, Config.PROJECTION_DIM)
        positive_indices = torch.zeros(B, B * 9, device=accelerator.device)
        
        batch_offsets = torch.arange(B, device=accelerator.device) * 9
        row_indices_broad = torch.arange(B, device=accelerator.device).unsqueeze(1)
        col_offsets = torch.arange(9, device=accelerator.device).unsqueeze(0)

        same_image_cols = batch_offsets.unsqueeze(1) + col_offsets
        positive_indices[row_indices_broad, same_image_cols] = 0.5

        global_positive_indices = local_indices + batch_offsets
        row_indices_flat = torch.arange(B, device=accelerator.device)
        positive_indices[row_indices_flat, global_positive_indices] = 0.95

        if useap:
            if fused_feats is not None:
                img_text_loss = 0.7 * info_nce_loss(
                    fused_feats, candidate_feats, positive_indices
                )
                img_text_loss += 0.3 * info_nce_loss(
                    anchor_feats, candidate_feats, positive_indices
                )
            else:
                # No text branch: fall back to image-only retrieval objective.
                img_text_loss = info_nce_loss(
                    anchor_feats, candidate_feats, positive_indices
                )
        else:
            scorer_input = fused_feats if fused_feats is not None else anchor_feats
            scores = model.scorer(scorer_input, candidate_feats)
            img_text_loss = F.cross_entropy(scores, positive_indices)

        batch_n = torch.tensor(float(query_imgs.shape[0]), device=accelerator.device)
        accu50_sum += accu_list[0].detach() * batch_n
        accu25_sum += accu_list[1].detach() * batch_n
        iou_sum += iou.detach() * batch_n
        info_nce_sum += img_text_loss.detach() * batch_n
        sample_count += batch_n

    reduced = accelerator.reduce(
        torch.stack([accu50_sum, accu25_sum, iou_sum, info_nce_sum, sample_count]),
        reduction="sum",
    )
    denom = reduced[4].clamp(min=1.0)

    return (
        (reduced[0] / denom).item(),
        (reduced[1] / denom).item(),
        (reduced[2] / denom).item(),
        (reduced[3] / denom).item(),
    )


# --- Training ---
def load_data_splits() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], set, set]:
    train_image_pairs = []
    test_image_pairs = []
    train_ids = set()
    test_ids = set()

    with open("/data/feihong/ckpt/train.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{Config.IMAGE_FOLDER}/{name}.png"
            train_image_pairs.append((query_path, search_path))
            train_ids.add(name)

    with open("/data/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{Config.IMAGE_FOLDER}/{name}.png"
            test_image_pairs.append((query_path, search_path))
            test_ids.add(name)

    return train_image_pairs, test_image_pairs, train_ids, test_ids


def train(save_path: str, end_num: float, use_ap: bool = True) -> None:
    valid_objectives = {"combined", "img_text_only", "bbox_only"}
    if Config.OPTIMIZE_OBJECTIVE not in valid_objectives:
        raise ValueError(
            f"Invalid OPTIMIZE_OBJECTIVE={Config.OPTIMIZE_OBJECTIVE}. "
            f"Choose from {sorted(valid_objectives)}."
        )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=Config.GRAD_ACCUMULATION_STEPS,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        torch.backends.cuda.matmul.allow_tf32 = Config.ENABLE_TF32
        torch.backends.cudnn.allow_tf32 = Config.ENABLE_TF32

    exp_name = save_path.split("/")[-1] if save_path else "default_exp"
    writer = SummaryWriter(f"runs/{exp_name}") if accelerator.is_main_process else None
    scaler = torch.amp.GradScaler(
        "cuda", enabled=Config.USE_AMP and accelerator.device.type == "cuda"
    )

    clearml_task = None
    clearml_logger = None
    if Config.CLEARML_ENABLED and accelerator.is_main_process:
        task_name = Config.CLEARML_TASK_NAME or exp_name
        clearml_task = Task.init(
            project_name=Config.CLEARML_PROJECT,
            task_name=task_name,
        )

        raw_config_dict = {
            key: getattr(Config, key)
            for key in dir(Config)
            if key.isupper() and not key.startswith("__")
        }
        config_dict = _to_clearml_serializable(raw_config_dict)
        clearml_task.connect(config_dict, name="config")

        clearml_logger = clearml_task.get_logger()
        clearml_logger.report_text(f"Experiment: {exp_name}")
        clearml_logger.report_text(
            f"Config: batch_size={Config.BATCH_SIZE}, lr={Config.LEARNING_RATE}, epochs={Config.NUM_EPOCHS}"
        )
        clearml_logger.report_text(
            "Full Config:\n" + json.dumps(config_dict, indent=2, sort_keys=True, ensure_ascii=False)
        )

    accelerator.print("Loading models and processor...")
    processor = AutoImageProcessor.from_pretrained(
        Config.MODEL_NAME, cache_dir=Config.CACHE_DIR
    )
    processor_sat = AutoImageProcessor.from_pretrained(
        Config.MODEL_NAME,
        cache_dir=Config.CACHE_DIR,
        size=Config.UNIV_SAT_SIZE,
    )
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # model = Encoder_heading(
    #     model_name=Config.MODEL_NAME, proj_dim=Config.PROJECTION_DIM
    # )
    model = Encoder_dino(
        proj_dim=Config.PROJECTION_DIM,
        usesg=True,
        useap=use_ap,
        freeze_vision_blocks=Config.FREEZE_VISION_BLOCKS,
        freeze_patch_generator=Config.FREEZE_PATCH_GENERATOR,
        freeze_cls_token=Config.FREEZE_CLS_TOKEN,
    )
    # model = Encoder_dino()

    anchors_full = get_tensor_anchors(accelerator.device)

    accelerator.print("Setting up dataset and dataloader...")
    train_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="train",
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=Config.BATCH_SIZE,
        **build_dataloader_kwargs(
            Config.NUM_WORKERS_TRAIN, drop_last=Config.DROP_LAST_TRAIN
        ),
    )

    test_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="test",
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=Config.BATCH_SIZE,
        **build_dataloader_kwargs(Config.NUM_WORKERS_VAL),
    )
    accelerator.print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=Config.LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.COSINE_EPOCHS
    )

    model, optimizer, scheduler, train_dataloader, test_dataloader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, test_dataloader
        )
    )

    accelerator.print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    soft_label = 0.3
    max_iou = 0
    min_info = 1000
    amp_enabled = Config.USE_AMP and accelerator.device.type == "cuda"

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_bbox_loss = 0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}",
            disable=not accelerator.is_main_process,
        )

        for i, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                target_pixel_values = batch["target_pixel_values"].to(
                    accelerator.device, non_blocking=True
                )
                search_pixel_values = batch["search_pixel_values"].to(
                    accelerator.device, non_blocking=True
                )
                input_ids = (
                    batch["input_ids"].to(accelerator.device, non_blocking=True)
                    if Config.USE_TEXT_INPUT
                    else None
                )
                local_indices = batch["index"].to(
                    accelerator.device, non_blocking=True
                )
                target_bbox = batch["bbox"].to(accelerator.device, non_blocking=True)
                angles = (
                    batch["angle"].to(accelerator.device, non_blocking=True)
                    if Config.USE_ANGLE_INPUT
                    else None
                )

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    (
                        pred_anchor,
                        _,
                        text_feats,
                        anchor_feats,
                        grid_feats,
                        fused_feats,
                    ) = model(target_pixel_values, search_pixel_values, input_ids, angles)

                    B = pred_anchor.shape[0]
                    pred_anchor = pred_anchor.view(
                        B, 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
                    )

                    new_gt_bbox, best_anchor_gi_gj = build_target(
                        target_bbox,
                        anchors_full,
                        (Config.UNIV_SAT_SIZE['width'], Config.UNIV_SAT_SIZE['height']),
                        (pred_anchor.shape[4], pred_anchor.shape[3]),
                    )

                    loss_geo, loss_cls = yolo_loss(
                        pred_anchor,
                        new_gt_bbox,
                        anchors_full,
                        best_anchor_gi_gj,
                        (Config.UNIV_SAT_SIZE['width'], Config.UNIV_SAT_SIZE['height']),
                    )
                    bbox_loss = loss_geo + loss_cls

                    candidate_feats = grid_feats.reshape(-1, Config.PROJECTION_DIM)
                    positive_indices = torch.zeros(B, B * 9, device=accelerator.device)

                    batch_offsets = torch.arange(B, device=accelerator.device) * 9
                    row_indices_broad = torch.arange(
                        B, device=accelerator.device
                    ).unsqueeze(1)
                    col_offsets = torch.arange(9, device=accelerator.device).unsqueeze(0)

                    same_image_cols = batch_offsets.unsqueeze(1) + col_offsets
                    positive_indices[row_indices_broad, same_image_cols] = soft_label

                    global_positive_indices = local_indices + batch_offsets
                    row_indices_flat = torch.arange(B, device=accelerator.device)
                    positive_indices[row_indices_flat, global_positive_indices] = 0.95


                    if fused_feats is not None:
                        img_text_loss = 0.7 * info_nce_loss(
                            fused_feats, candidate_feats, positive_indices
                        )
                        img_text_loss += 0.3 * info_nce_loss(
                            anchor_feats, candidate_feats, positive_indices
                        )
                    else:
                        # No text branch: fall back to image-only retrieval objective.
                        img_text_loss = info_nce_loss(
                            anchor_feats, candidate_feats, positive_indices
                        )


                    scheduled_bbox_weight, scheduled_retrieval_weight = get_loss_weights(
                        epoch, Config.COSINE_EPOCHS, end_num
                    )
                    if Config.OPTIMIZE_OBJECTIVE == "combined":
                        bbox_weight = scheduled_bbox_weight
                        retrieval_weight = scheduled_retrieval_weight
                        loss = retrieval_weight * img_text_loss + bbox_weight * bbox_loss
                    elif Config.OPTIMIZE_OBJECTIVE == "img_text_only":
                        bbox_weight = 0.0
                        retrieval_weight = 1.0
                        loss = img_text_loss
                    else:  # bbox_only
                        bbox_weight = 1.0
                        retrieval_weight = 0.0
                        loss = bbox_loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()
            progress_bar.set_postfix(
                {
                    "bbox_loss": f"{bbox_loss.item():.4f}",
                    "img_text_loss": f"{img_text_loss.item():.4f}",
                    "retrieval_weight": f"{retrieval_weight:.4f}",
                    "bbox_weight": f"{bbox_weight:.4f}",
                }
            )

            if clearml_logger is not None:
                global_step = epoch * len(train_dataloader) + i
                clearml_logger.report_scalar("train", "loss", loss.item(), global_step)
                clearml_logger.report_scalar(
                    "train", "bbox_loss", bbox_loss.item(), global_step
                )
                clearml_logger.report_scalar(
                    "train", "img_text_loss", img_text_loss.item(), global_step
                )

        scheduler.step()

        if epoch % 2 == 0:
            os.makedirs(save_path, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            save_filename = f"{save_path}/best_iou.pth"

            accelerator.save(unwrapped_model.state_dict(), save_filename)

            # accelerator.print(f"Running validation at epoch {epoch + 1}...")
            # accu50, accu25, iou, info_nce = validate(
            #     test_dataloader,
            #     model,
            #     accelerator,
            #     anchors_full,
            #     (Config.UNIV_SAT_SIZE[0], Config.UNIV_SAT_SIZE[1]),
            #     useap=use_ap,
            # )
            # if accelerator.is_main_process:
            #     accelerator.print(
            #         f"Val Epoch {epoch + 1}: Accu@50={accu50:.4f}, Accu@25={accu25:.4f}, IoU={iou:.4f}, "
            #         f"InfoNCE={info_nce:.4f}"
            #     )

            #     if clearml_logger is not None:
            #         clearml_logger.report_scalar("validation", "accu50", accu50, epoch)
            #         clearml_logger.report_scalar("validation", "accu25", accu25, epoch)
            #         clearml_logger.report_scalar("validation", "iou", iou, epoch)
            #         clearml_logger.report_scalar("validation", "info_nce", info_nce, epoch)

            #     if iou > max_iou:
            #         max_iou = iou
            #         os.makedirs(save_path, exist_ok=True)
            #         unwrapped_model = accelerator.unwrap_model(model)
            #         save_filename = f"{save_path}/best_iou.pth"
            #         accelerator.save(unwrapped_model.state_dict(), save_filename)
            #         accelerator.print(f"Saved best IoU checkpoint: {save_filename}")

        if clearml_logger is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            clearml_logger.report_scalar("lr", "learning_rate", current_lr, epoch)

    accelerator.print("Training complete.")
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        save_filename = f"{save_path}/last.pth"
        accelerator.save(unwrapped_model.state_dict(), save_filename)
        print(f"Saved checkpoint: {save_filename}")
        if clearml_task is not None:
            clearml_task.close()
    if writer is not None:
        writer.close()



# --- Main ---
if __name__ == "__main__":
    end_num = 0.1
    exp_name = f"train_dino_"
    save_dir = f"/data/feihong/ckpt/{exp_name}"

    if os.path.exists(save_dir):
        print(f"Experiment directory '{save_dir}' already exists.")
        print("Re-using directory. Logs and checkpoints might be overwritten.")
    else:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created experiment directory: {save_dir}")

    train(save_dir, end_num, use_ap=True)
