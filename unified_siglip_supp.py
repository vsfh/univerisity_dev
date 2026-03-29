# --- Configuration ---
import os
import json
import random
import time
from glob import glob
from typing import Dict, List, Tuple, Optional

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

from model import Encoder_heading, Encoder_abla
from bbox.yolo_utils import (
    get_tensor_anchors,
    build_target,
    yolo_loss,
    eval_iou_acc,
)

cudnn.benchmark = True


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
    UNIV_SAT_SIZE = (640, 640)
    DRONE_SIZE = (256, 256)

    NUM_EPOCHS = 60
    BATCH_SIZE = 10
    GRAD_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 5e-5
    LR_MIN = 1e-10
    COSINE_EPOCHS = 60
    BBOX_LOSS_WEIGHT = 0.1
    HEADING_LOSS_WEIGHT = 0.01
    PROJECTION_DIM = 768

    HEADING_TO_TARGET = {
        0: [0.0, 0.0],
        90: [0.0, 1.0],
        180: [1.0, 0.0],
        270: [1.0, 1.0],
    }

    TARGET_TO_HEADING = {
        (0.0, 0.0): 0,
        (0.0, 1.0): 90,
        (1.0, 0.0): 180,
        (1.0, 1.0): 270,
    }


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
    ratio = image.size[0] / target_size[0]
    image = image.resize(target_size)

    new_x1 = (x1 - left) / ratio
    new_y1 = (y1 - top) / ratio
    new_x2 = (x2 - left) / ratio
    new_y2 = (y2 - top) / ratio

    return image, [new_x1, new_y1, new_x2, new_y2]


def resize_drone_image(
    image: Image.Image, target_size: Tuple[int, int] = Config.DRONE_SIZE
) -> Image.Image:
    return image.resize(target_size, Image.Resampling.BILINEAR)


def visualize_batch(
    batch: Dict, save_dir: str = "runs/visualizations", max_samples: int = 6
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    target_pixels = batch["target_pixel_values"]
    search_pixels = batch["search_pixel_values"]
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
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{names[i]}_combined.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(
        f"Saved {min(target_pixels.shape[0], max_samples)} visualizations to {save_dir}"
    )


# --- Dataset ---
class GeoDataset(Dataset):
    def __init__(
        self,
        image_pairs: List[Tuple[str, str]],
        processor,
        processor_sat,
        tokenizer,
        img_to_text_dict: Dict,
        bbox_dict: Optional[Dict] = None,
        mode: str = "train",
    ):
        self.image_paths = image_pairs
        self.processor = processor
        self.processor_sat = processor_sat
        self.tokenizer = tokenizer
        self.img_to_text_dict = img_to_text_dict
        self.bbox_dict = bbox_dict
        self.mode = mode
        self.default_bbox = (1536, 656, 2268, 1374)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        query_path, search_path = self.image_paths[idx]
        heading = 0
        name = query_path.split("/")[-2]

        if self.mode == "train":
            choices = []
            for number in ["01.", "02.", "03.", "04.", "05."]:
                new_query_path = query_path.replace("01.", number)
                if os.path.exists(new_query_path):
                    choices.append(new_query_path)
            if choices:
                query_path = random.sample(choices, 1)[0]

        text_name = name + "_01"
        text_description = self.img_to_text_dict.get(text_name, "")

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
            "Description",
            "()",
        ]:
            text_description = text_description.replace(noun, "")

        try:
            query_image = Image.open(query_path).convert("RGB")
            search_image = Image.open(search_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping this item.")
            return self.__getitem__((idx + 1) % len(self))

        query_image = resize_drone_image(query_image)

        bbox_key = name
        bbox = self.bbox_dict.get(bbox_key, None) if self.bbox_dict else None
        if bbox is None:
            bbox = self.default_bbox

        augmented_crop_image, normalized_bbox = format_satellite_img_bbox(
            search_image, bbox, mode=self.mode, target_size=Config.UNIV_SAT_SIZE
        )

        center_x = (
            (normalized_bbox[0] + normalized_bbox[2]) / 2 / Config.UNIV_SAT_SIZE[0]
        )
        center_y = (
            (normalized_bbox[1] + normalized_bbox[3]) / 2 / Config.UNIV_SAT_SIZE[0]
        )

        col_idx = min(int(center_x * 3), 2)
        row_idx = min(int(center_y * 3), 2)
        index = row_idx * 3 + col_idx

        heading_target = torch.tensor(
            Config.HEADING_TO_TARGET[heading], dtype=torch.float32
        )

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
            "heading": heading_target,
            "name": name,
        }


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
    img_size: int,
    useap: bool = True,
) -> Tuple[float, float, float, float, float]:
    model.eval()

    accu50_meter = AverageMeter()
    accu25_meter = AverageMeter()
    iou_meter = AverageMeter()
    info_nce_meter = AverageMeter()
    heading_meter = AverageMeter()

    for batch in tqdm(loader, desc="Validating"):
        query_imgs = batch["target_pixel_values"]
        rs_imgs = batch["search_pixel_values"]
        ori_gt_bbox = batch["bbox"]
        input_ids = batch["input_ids"]
        local_indices = batch["index"]
        heading_target = batch["heading"]

        with torch.no_grad():
            (
                pred_anchor,
                pred_heading,
                text_feats,
                anchor_feats,
                grid_feats,
                fused_feats,
            ) = model(query_imgs, rs_imgs, input_ids)

        B = pred_anchor.shape[0]
        pred_anchor = pred_anchor.view(
            B, 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
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
            img_text_loss = 0.7 * info_nce_loss(
                fused_feats, candidate_feats, positive_indices
            )
            img_text_loss += 0.3 * info_nce_loss(
                anchor_feats, candidate_feats, positive_indices
            )
        else:
            scores = model.scorer(fused_feats, candidate_feats)
            img_text_loss = F.cross_entropy(scores, positive_indices)

        heading_loss = F.mse_loss(pred_heading, heading_target)

        accu50_meter.update(accu_list[0].item(), query_imgs.shape[0])
        accu25_meter.update(accu_list[1].item(), query_imgs.shape[0])
        iou_meter.update(iou.item(), query_imgs.shape[0])
        info_nce_meter.update(img_text_loss.item(), query_imgs.shape[0])
        heading_meter.update(heading_loss.item(), query_imgs.shape[0])

    return (
        accu50_meter.avg,
        accu25_meter.avg,
        iou_meter.avg,
        info_nce_meter.avg,
        heading_meter.avg,
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
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=Config.GRAD_ACCUMULATION_STEPS,
        kwargs_handlers=[ddp_kwargs],
    )

    exp_name = save_path.split("/")[-1] if save_path else "default_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    clearml_task = None
    clearml_logger = None
    if Config.CLEARML_ENABLED and accelerator.is_main_process:
        task_name = Config.CLEARML_TASK_NAME or exp_name
        clearml_task = Task.init(
            project_name=Config.CLEARML_PROJECT,
            task_name=task_name,
        )
        clearml_logger = clearml_task.get_logger()
        clearml_logger.report_text(f"Experiment: {exp_name}")
        clearml_logger.report_text(
            f"Config: batch_size={Config.BATCH_SIZE}, lr={Config.LEARNING_RATE}, epochs={Config.NUM_EPOCHS}"
        )

    print("Loading data splits...")
    train_image_pairs, test_image_pairs, train_ids, test_ids = load_data_splits()
    print(f"Train: {len(train_image_pairs)}, Test: {len(test_image_pairs)}")

    print("Loading models and processor...")
    processor = AutoImageProcessor.from_pretrained(
        Config.MODEL_NAME, cache_dir=Config.CACHE_DIR
    )
    processor_sat = AutoImageProcessor.from_pretrained(
        Config.MODEL_NAME,
        cache_dir=Config.CACHE_DIR,
        size={"height": 640, "width": 640},
    )
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # model = Encoder_heading(
    #     model_name=Config.MODEL_NAME, proj_dim=Config.PROJECTION_DIM
    # )
    model = Encoder_abla(
        model_name=Config.MODEL_NAME,
        proj_dim=Config.PROJECTION_DIM,
        usesg=True,
        useap=use_ap,
    )

    img_to_text_dict = json.load(open(Config.TEXT_FILE, "r"))
    train_bbox_dict = json.load(open(Config.TRAIN_BBOX_FILE, "r"))
    test_bbox_dict = json.load(open(Config.TEST_BBOX_FILE, "r"))

    anchors_full = get_tensor_anchors(accelerator.device)

    print("Setting up dataset and dataloader...")
    train_dataset = GeoDataset(
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
        batch_size=Config.BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
    )

    test_dataset = GeoDataset(
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
        batch_size=Config.BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.COSINE_EPOCHS
    )

    model, optimizer, scheduler, train_dataloader, test_dataloader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, test_dataloader
        )
    )

    accelerator.print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    soft_label = 0.5
    max_iou = 0
    min_info = 1000

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_bbox_loss = 0
        total_heading_loss = 0
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}"
        )

        for i, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                target_pixel_values = batch["target_pixel_values"]
                search_pixel_values = batch["search_pixel_values"]
                input_ids = batch["input_ids"]
                local_indices = batch["index"]
                target_bbox = batch["bbox"]
                heading_target = batch["heading"]

                (
                    pred_anchor,
                    pred_heading,
                    text_feats,
                    anchor_feats,
                    grid_feats,
                    fused_feats,
                ) = model(target_pixel_values, search_pixel_values, input_ids)

                B = pred_anchor.shape[0]
                pred_anchor = pred_anchor.view(
                    B, 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
                )

                new_gt_bbox, best_anchor_gi_gj = build_target(
                    target_bbox,
                    anchors_full,
                    Config.UNIV_SAT_SIZE[0],
                    pred_anchor.shape[3],
                )

                loss_geo, loss_cls = yolo_loss(
                    pred_anchor,
                    new_gt_bbox,
                    anchors_full,
                    best_anchor_gi_gj,
                    Config.UNIV_SAT_SIZE[0],
                )
                bbox_loss = loss_geo + loss_cls
                heading_loss = F.mse_loss(pred_heading, heading_target)

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

                if use_ap:
                    img_text_loss = 0.7 * info_nce_loss(
                        fused_feats, candidate_feats, positive_indices
                    )
                    img_text_loss += 0.3 * info_nce_loss(
                        anchor_feats, candidate_feats, positive_indices
                    )
                else:
                    scores = model.scorer(fused_feats, candidate_feats)
                    img_text_loss = F.cross_entropy(scores, positive_indices)

                bbox_weight, retrieval_weight = get_loss_weights(
                    epoch, Config.COSINE_EPOCHS, end_num
                )
                loss = (
                    retrieval_weight * img_text_loss
                    + bbox_weight * bbox_loss
                    # + heading_loss * Config.HEADING_LOSS_WEIGHT
                )

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()
            total_heading_loss += heading_loss.item()
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

        if (epoch + 1) % 10 == 0 and accelerator.is_main_process:
            accelerator.print(f"Running validation at epoch {epoch + 1}...")
            accu50, accu25, iou, info_nce, heading = validate(
                test_dataloader,
                model,
                accelerator,
                anchors_full,
                Config.UNIV_SAT_SIZE[0],
                useap=use_ap,
            )
            accelerator.print(
                f"Val Epoch {epoch + 1}: Accu@50={accu50:.4f}, Accu@25={accu25:.4f}, IoU={iou:.4f}, "
                f"InfoNCE={info_nce:.4f}, Heading={heading:.4f}"
            )
            
            if clearml_logger is not None:
                clearml_logger.report_scalar("validation", "accu50", accu50, epoch)
                clearml_logger.report_scalar("validation", "accu25", accu25, epoch)
                clearml_logger.report_scalar("validation", "iou", iou, epoch)
                clearml_logger.report_scalar("validation", "info_nce", info_nce, epoch)
                clearml_logger.report_scalar("validation", "heading", heading, epoch)

            if iou > max_iou:
                max_iou = iou
                os.makedirs(save_path, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                save_filename = f"{save_path}/best_iou.pth"
                accelerator.save(unwrapped_model.state_dict(), save_filename)
                accelerator.print(f"Saved best IoU checkpoint: {save_filename}")

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
    writer.close()


# --- Evaluation ---
def pred_to_heading(pred: np.ndarray) -> int:
    min_dist = float("inf")
    best_heading = 0
    for (t0, t1), heading in Config.TARGET_TO_HEADING.items():
        dist = (pred[0] - t0) ** 2 + (pred[1] - t1) ** 2
        if dist < min_dist:
            min_dist = dist
            best_heading = heading
    return best_heading


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm[..., None]).flatten()


def compute_scorer_similarity(
    query_feat: np.ndarray,
    candidate_feats: np.ndarray,
    scorer: nn.Module,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute similarity scores using the scorer model.
    
    Args:
        query_feat: Query feature array of shape (768,)
        candidate_feats: Candidate feature array of shape (N, 768) or (9, 768)
        scorer: ScoreHead model
        device: Device to run computation on
    
    Returns:
        scores: Array of shape (N,) with similarity scores
    """
    scorer.eval()
    
    query_tensor = torch.from_numpy(query_feat).float().unsqueeze(0).to(device)
    candidate_tensor = torch.from_numpy(candidate_feats).float().to(device)
    
    with torch.no_grad():
        scores = scorer(query_tensor, candidate_tensor)
    
    return scores.cpu().numpy().flatten()


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_extra: bool = False,
    use_ap: bool = True,
) -> Tuple[Dict, Dict]:
    model.eval()
    res_search = {}
    res_fused_query = {}

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
        for batch in tqdm(loader):
            if batch is None:
                continue

            names = batch["name"]
            query_inputs = batch["target_pixel_values"].to(device, non_blocking=True)
            search_inputs = batch["search_pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)

            (
                pred_anchor,
                pred_heading,
                text_feats,
                anchor_feats,
                grid_feats,
                fused_feats,
            ) = model(query_inputs, search_inputs, input_ids)

            grid_feats_np = grid_feats.float().cpu().numpy()
            fused_feats_np = fused_feats.float().cpu().numpy()
            # fused_feats_np = anchor_feats.float().cpu().numpy()

            for i, name in enumerate(names):
                res_search[name] = grid_feats_np[i]
                res_fused_query[name] = fused_feats_np[i]

    if use_extra:
        satellite_files = sorted(
            [f for f in os.listdir(Config.SATELLITE_FOLDER) if f.endswith(".png")]
        )
        print(f"Processing {len(satellite_files)} satellite images...")

        processor_sat = AutoImageProcessor.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.CACHE_DIR,
            size={"height": 640, "width": 640},
        )

        satellite_batch = []
        satellite_names = []
        for sat_file in tqdm(satellite_files):
            sat_name = sat_file.replace(".png", "")
            sat_path = os.path.join(Config.SATELLITE_FOLDER, sat_file)
            try:
                sat_image = Image.open(sat_path).convert("RGB")
                sat_image = sat_image.crop((840, 0, 3000, 2160)).resize((640, 640))
                sat_pixels = processor_sat(images=sat_image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                satellite_batch.append(sat_pixels)
                satellite_names.append(sat_name)
            except Exception as e:
                print(f"Error processing {sat_file}: {e}")
                continue

        for i in range(0, len(satellite_batch), Config.BATCH_SIZE):
            batch_pixels = torch.stack(satellite_batch[i : i + Config.BATCH_SIZE]).to(
                device
            )
            batch_names = satellite_names[i : i + Config.BATCH_SIZE]

            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
                grid_feats = model.ref_forward(batch_pixels)

            grid_feats_np = grid_feats.float().cpu().numpy()
            for j, name in enumerate(batch_names):
                res_search[name] = grid_feats_np[j]

    return res_search, res_fused_query


def evaluate_retrieval(
    search_res: np.ndarray, 
    fused_query_res: np.ndarray, 
    test_num: int = 100,
    scorer: nn.Module = None,
    device: str = "cuda",
) -> Tuple[int, int, int]:
    distances = json.load(open("/data/feihong/ckpt/distances.json", "r"))

    eval_keys = [str(k) for k in search_res.keys()]
    unique_names = [k for k in eval_keys if "new" not in k]

    top1, top5, top10 = 0, 0, 0

    for key in tqdm(unique_names):
        fused_query_feature = fused_query_res[key]

        ex_img_list = random.sample(
            unique_names, min(test_num - 1, len(unique_names) - 1)
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

        res = []
        for img_name in ex_img_list:
            if len(search_res[img_name].shape) == 1:
                grid_fea = search_res[img_name][None]
            else:
                grid_fea = search_res[img_name]
            
            if scorer is not None:
                sim_scores = compute_scorer_similarity(
                    fused_query_feature, grid_fea, scorer, device
                )
            else:
                sim_scores = compute_cosine_similarity(fused_query_feature, grid_fea)
            
            res.append(sim_scores)

        img_res = np.array(res).max(1).argsort()[-15:][::-1]

        if any(cand in candidate_indices for cand in img_res[:1]):
            top1 += 1
        if any(cand in candidate_indices for cand in img_res[:5]):
            top5 += 1
        if any(cand in candidate_indices for cand in img_res[:10]):
            top10 += 1

    print(f"Retrieval Metrics:")
    print(
        f"Top 1: {top1} / {len(unique_names)} ({top1 / len(unique_names) * 100:.2f}%)"
    )
    print(
        f"Top 5: {top5} / {len(unique_names)} ({top5 / len(unique_names) * 100:.2f}%)"
    )
    print(
        f"Top 10: {top10} / {len(unique_names)} ({top10 / len(unique_names) * 100:.2f}%)"
    )

    return top1, top5, top10


def run_evaluation(end_num: float, run: bool = True, use_extra: bool = False) -> None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 43
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if run:
        print("Initializing Evaluation...")
        img_to_text_dict = json.load(open(Config.TEXT_FILE, "r"))

        processor = AutoImageProcessor.from_pretrained(
            Config.MODEL_NAME, cache_dir=Config.CACHE_DIR
        )
        processor_sat = AutoImageProcessor.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.CACHE_DIR,
            size={"height": 640, "width": 640},
        )
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        test_bbox_dict = json.load(open(Config.TEST_BBOX_FILE, "r"))

        # model = Encoder_heading(
        #     model_name=Config.MODEL_NAME, proj_dim=Config.PROJECTION_DIM
        # ).to(DEVICE)
        model = Encoder_abla(
            model_name=Config.MODEL_NAME,
            proj_dim=Config.PROJECTION_DIM,
            usesg=True,
            useap=True,
        ).to(DEVICE)
        model_path = f"/data/feihong/ckpt/supp_{end_num}/last.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded checkpoint: {model_path}")
        else:
            print(f"Checkpoint not found: {model_path}")
            return
        model.eval()

        eval_image_pairs = []
        with open("/data/feihong/ckpt/test.txt", "r") as f:
            for line in f:
                query_path = line.strip()
                index = query_path.split("/")[-2]
                search_path = f"{Config.IMAGE_FOLDER}/{index}.png"
                if os.path.exists(query_path) and os.path.exists(search_path):
                    eval_image_pairs.append((query_path, search_path))

        print(f"Eval pairs: {len(eval_image_pairs)}")

        dataset = GeoDataset(
            eval_image_pairs,
            processor,
            processor_sat,
            tokenizer,
            img_to_text_dict,
            test_bbox_dict,
            mode="test",
        )

        def collate_fn(batch):
            batch = [b for b in batch if b is not None]
            if len(batch) == 0:
                return None
            return torch.utils.data.dataloader.default_collate(batch)

        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        res_search, res_fused_query = extract_features(model, loader, DEVICE, use_extra)

        np.savez("eval_search_heading.npz", **res_search)
        np.savez("eval_fused_query_heading.npz", **res_fused_query)
        print("Evaluation feature extraction complete.")
        
        # evaluate_retrieval(res_search, res_fused_query, scorer=model.scorer, device=DEVICE)
    else:
        # model = Encoder_abla(
        #     model_name=Config.MODEL_NAME,
        #     proj_dim=Config.PROJECTION_DIM,
        #     usesg=True,
        #     useap=False,
        # ).to(DEVICE)
        # model_path = f"/data/feihong/ckpt/supp_{end_num}/last.pth"
        # if os.path.exists(model_path):
        #     model.load_state_dict(torch.load(model_path, map_location="cpu"))
        #     print(f"Loaded checkpoint: {model_path}")
        # else:
        #     print(f"Checkpoint not found: {model_path}")
        #     return
        # model.eval()
        search_res = np.load("eval_search_heading.npz", allow_pickle=True)
        fused_query_res = np.load("eval_fused_query_heading.npz", allow_pickle=True)
        evaluate_retrieval(search_res, fused_query_res, device=DEVICE)


# --- Main ---
if __name__ == "__main__":
    end_num = 0.1
    exp_name = f"supp_{end_num}"
    save_dir = f"/data/feihong/ckpt/{exp_name}"

    if os.path.exists(save_dir):
        print(f"Experiment directory '{save_dir}' already exists.")
        print("Re-using directory. Logs and checkpoints might be overwritten.")
    else:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created experiment directory: {save_dir}")

    # train(save_dir, end_num, use_ap=True)
    run_evaluation(end_num, run=True, use_extra=False)
    run_evaluation(end_num, run=False, use_extra=False)
