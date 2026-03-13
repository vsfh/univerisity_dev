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
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
import time
from model import Encoder_heading, Encoder_abla
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


MODEL_NAME = "google/siglip-base-patch16-224"
CACHE_DIR = "/data/feihong/hf_cache"
DRONE_VIEW_FOLDER = "/data/feihong/drone_view"
IMAGE_FOLDER = "/data/feihong/image_1024"
HEADING_FOLDER = "/data/feihong/range_250"
TEXT_FILE = "/data/feihong/drone_text_single_long.json"
TRAIN_BBOX_FILE = "/data/feihong/univerisity_dev/runs/train.json"
TEST_BBOX_FILE = "/data/feihong/univerisity_dev/runs/test.json"

SAT_ORIG_SIZE = (3840, 2160)
UNIV_SAT_SIZE = (640, 640)
DRONE_SIZE = (256, 256)

NUM_EPOCHS = 30
BATCH_SIZE = 15
GRAD_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
LR_MIN = 1e-10
COSINE_EPOCHS = 30
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


def get_loss_weights(epoch, num_epochs, end_num):
    progress = epoch / num_epochs
    bbox_weight = (end_num - 0.5) * progress + 0.5
    retrieval_weight = 1.0 - bbox_weight
    return bbox_weight, retrieval_weight


def format_satellite_img_bbox(
    image,
    bbox,
    mode="train",
    target_size=UNIV_SAT_SIZE,
):
    x1, y1, x2, y2 = bbox
    width, height = image.size

    min_crop = max(y2 - y1, x2 - x1) * 1.2
    crop_size = random.uniform(min_crop, max(min_crop, height))
    if mode == "test":
        crop_size = height

    min_left = 0
    max_left = width - crop_size
    min_top = 0
    max_top = height - crop_size

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


def visualize(batch, save_dir="runs/visualizations", max_samples=6):
    """
    Visualize and save target (drone) and search (satellite) images from a batch.

    Args:
        batch: Dictionary containing:
            - target_pixel_values: Processed drone images (B, C, H, W)
            - search_pixel_values: Processed satellite images (B, C, H, W)
            - name: List of image names
        save_dir: Directory to save visualizations
        max_samples: Maximum number of samples to visualize per batch
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    target_pixels = batch["target_pixel_values"]  # (B, C, H, W)
    search_pixels = batch["search_pixel_values"]  # (B, C, H, W)
    names = batch.get("name", [f"sample_{i}" for i in range(target_pixels.shape[0])])

    for i in range(min(target_pixels.shape[0], max_samples)):
        # Denormalize: SigLIP uses mean=0.5, std=0.5
        # Reverse: img * std + mean = img * 0.5 + 0.5

        # Target (drone) image
        target_img = target_pixels[i].cpu().numpy()
        target_img = np.transpose(target_img, (1, 2, 0))  # (H, W, C)
        target_img = (target_img * 0.5 + 0.5).clip(0, 1)

        # Search (satellite) image
        search_img = search_pixels[i].cpu().numpy()
        search_img = np.transpose(search_img, (1, 2, 0))
        search_img = (search_img * 0.5 + 0.5).clip(0, 1)

        # Create combined visualization with header
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


class GeoDataset(Dataset):
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
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        query_path, search_path = self.image_paths[idx]

        heading = 0

        name = query_path.split("/")[-2]
        choice = []
        if self.mode == "train":
            for number in [
                "01.",
                "02.",
                "03.",
                "04.",
                "05.",
            ]:
                new_query_path = query_path.replace("01.", number)
                if not os.path.exists(new_query_path):
                    continue
                choice.append(new_query_path)

            if choice:
                query_path = random.sample(choice, 1)[0]
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
        # score = random.random()
        # if score > 0.5:
        #     crop_length = int(150 * (score - 0.5))
        #     query_image = query_image.crop(
        #         (crop_length, crop_length, 1080 - crop_length, 1080 - crop_length)
        #     )

        query_image = resize_drone_image(query_image)

        bbox_key = name
        bbox = None
        if self.bbox_dict:
            bbox = self.bbox_dict.get(bbox_key, None)
        if bbox is None:
            bbox = (1536, 656, 2268, 1374)

        augmented_crop_image, normalized_bbox = format_satellite_img_bbox(
            search_image, bbox, mode=self.mode, target_size=UNIV_SAT_SIZE
        )

        center_x = (normalized_bbox[0] + normalized_bbox[2]) / 2 / UNIV_SAT_SIZE[0]
        center_y = (normalized_bbox[1] + normalized_bbox[3]) / 2 / UNIV_SAT_SIZE[0]

        col_idx = min(int(center_x * 3), 2)
        row_idx = min(int(center_y * 3), 2)
        index = row_idx * 3 + col_idx

        heading_target = torch.tensor(HEADING_TO_TARGET[heading], dtype=torch.float32)

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


def info_nce_loss(query_feats, candidate_feats, positive_indices, temperature=0.07):
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature
    return F.cross_entropy(sim_matrix, positive_indices)


def validation(loader, model, accelerator, anchors_full, img_size, useap=True):
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
        if useap:
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
        else:
            query_feats = F.normalize(fused_feats, p=2, dim=1)
            candidate_feats = F.normalize(grid_feats, p=2, dim=1)

            # Similarity matrix: (B, B)
            sim_matrix = torch.matmul(query_feats, candidate_feats.T)
            # Positive indices: diagonal (each query's positive is itself)
            positive_indices = torch.arange(B, device=query_feats.device)

            img_text_loss = F.cross_entropy(sim_matrix, positive_indices)

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


def main(save_path, end_num):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        kwargs_handlers=[ddp_kwargs],
    )

    not_update = 0
    soft_label = 0.5
    exp_name = save_path.split("/")[-1] if save_path else "default_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Reading train/test IDs from files...")
    train_ids = set()

    train_image_pairs = []
    test_image_pairs = []
    with open("/data/feihong/ckpt/train.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            train_image_pairs.append((query_path, search_path))
            train_ids.add(name)

    test_ids = set()
    with open("/data/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            test_image_pairs.append((query_path, search_path))
            test_ids.add(name)

    print(f"Train IDs: {len(train_image_pairs)}, Test IDs: {len(test_image_pairs)}")

    print(f"Train: {len(train_image_pairs)}, Test: {len(test_image_pairs)}")

    print("Loading models and processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor_sat = AutoImageProcessor.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    useap = True
    model = Encoder_heading(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM)
    # useap = False
    # model = Encoder_abla(
    #     model_name=MODEL_NAME, proj_dim=PROJECTION_DIM, usesg=True, useap=useap
    # )

    img_to_text_dict = json.load(open(TEXT_FILE, "r"))
    train_bbox_dict = json.load(open(TRAIN_BBOX_FILE, "r"))
    test_bbox_dict = json.load(open(TEST_BBOX_FILE, "r"))

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
        batch_size=BATCH_SIZE,
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
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
    )

    bbox_params = []
    retrieval_params = []
    for name, param in model.named_parameters():
        if any(
            x in name
            for x in [
                "bbox_transformer",
                "bbox_fcn_out",
                "bbox_adapter",
                "bbox_heading",
            ]
        ):
            bbox_params.append(param)
        else:
            retrieval_params.append(param)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_EPOCHS
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
        total_heading_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

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
                heading_loss = F.mse_loss(pred_heading, heading_target)

                if useap:
                    candidate_feats = grid_feats.reshape(-1, PROJECTION_DIM)
                    positive_indices = torch.zeros(B, B * 9, device=accelerator.device)

                    batch_offsets = torch.arange(B, device=accelerator.device) * 9
                    row_indices_broad = torch.arange(
                        B, device=accelerator.device
                    ).unsqueeze(1)
                    col_offsets = torch.arange(9, device=accelerator.device).unsqueeze(
                        0
                    )

                    same_image_cols = batch_offsets.unsqueeze(1) + col_offsets
                    positive_indices[row_indices_broad, same_image_cols] = soft_label

                    global_positive_indices = local_indices + batch_offsets
                    row_indices_flat = torch.arange(B, device=accelerator.device)
                    positive_indices[row_indices_flat, global_positive_indices] = 0.95

                    img_text_loss = 0.7 * info_nce_loss(
                        fused_feats, candidate_feats, positive_indices
                    )
                    img_text_loss += 0.3 * info_nce_loss(
                        anchor_feats, candidate_feats, positive_indices
                    )

                else:
                    query_feats = F.normalize(fused_feats, p=2, dim=1)
                    candidate_feats = F.normalize(grid_feats, p=2, dim=1)

                    # Similarity matrix: (B, B)
                    sim_matrix = torch.matmul(query_feats, candidate_feats.T)
                    # Positive indices: diagonal (each query's positive is itself)
                    positive_indices = torch.arange(B, device=query_feats.device)

                    img_text_loss = F.cross_entropy(sim_matrix, positive_indices)

                bbox_weight, retrieval_weight = get_loss_weights(
                    epoch, COSINE_EPOCHS, end_num
                )
                loss = (
                    retrieval_weight * img_text_loss
                    + bbox_weight * bbox_loss
                    + heading_loss * HEADING_LOSS_WEIGHT
                )

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()
            total_heading_loss += heading_loss.item()
            progress_bar.set_postfix(
                {
                    # "loss": f"{loss.item():.4f}",
                    "bbox_loss": f"{bbox_loss.item():.4f}",
                    # "heading_loss": f"{heading_loss.item():.4f}",
                    "img_text_loss": f"{img_text_loss.item():.4f}",
                    "retrieval_weight": f"{retrieval_weight:.4f}",
                    "bbox_weight": f"{bbox_weight:.4f}",
                }
            )
            writer.add_scalar(
                "Loss/train_batch", loss.item(), epoch * len(train_dataloader) + i
            )

        avg_loss = total_loss / len(train_dataloader)
        avg_bbox_loss = total_bbox_loss / len(train_dataloader)
        avg_heading_loss = total_heading_loss / len(train_dataloader)
        # model.eval()
        # accu50, accu25, val_iou, val_info_nce, val_heading = validation(
        #     test_dataloader,
        #     model,
        #     accelerator,
        #     anchors_full,
        #     UNIV_SAT_SIZE[0],
        #     useap=useap,
        # )
        # writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        # writer.add_scalar("Metrics/val_iou", val_iou, epoch)
        # writer.add_scalar("Metrics/val_accu50", accu50, epoch)
        # writer.add_scalar("Metrics/val_accu25", accu25, epoch)
        # writer.add_scalar("Metrics/val_heading", val_heading, epoch)
        # accelerator.print(
        #     f"Epoch {epoch + 1} finished. "
        #     f"Avg train Loss: {avg_loss:.4f} (bbox: {avg_bbox_loss:.4f}, heading: {avg_heading_loss:.4f}). "
        #     f"IoU: {val_iou:.4f}, Accu50: {accu50:.4f}, Accu25: {accu25:.4f}, "
        #     f"InfoNCE: {val_info_nce:.4f}, Heading: {val_heading:.4f}"
        # )

        # if accelerator.is_main_process:
        #     if val_info_nce < min_info:
        #         os.makedirs(save_path, exist_ok=True)
        #         unwrapped_model = accelerator.unwrap_model(model)

        #         save_filename = f"{save_path}/best_info_{not_update}.pth"
        #         accelerator.save(unwrapped_model.state_dict(), save_filename)

        #         print(f"Saved checkpoint info: {save_filename}")
        #         min_info = val_info_nce
        #         not_update += 1
        #     if max_iou < val_iou:
        #         os.makedirs(save_path, exist_ok=True)
        #         unwrapped_model = accelerator.unwrap_model(model)

        #         save_filename = f"{save_path}/best_iou_{not_update}.pth"
        #         accelerator.save(unwrapped_model.state_dict(), save_filename)

        #         print(f"Saved checkpoint iou: {save_filename}")
        #         max_iou = val_iou
        #         not_update += 1
        scheduler.step()

    accelerator.print("Training complete.")
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)

        save_filename = f"{save_path}/last.pth"
        accelerator.save(unwrapped_model.state_dict(), save_filename)

        print(f"Saved checkpoint info: {save_filename}")
    writer.close()


def pred_to_heading(pred):
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    min_dist = float("inf")
    best_heading = 0
    for (t0, t1), heading in TARGET_TO_HEADING.items():
        dist = (pred[0] - t0) ** 2 + (pred[1] - t1) ** 2
        if dist < min_dist:
            min_dist = dist
            best_heading = heading
    return best_heading


def eval(end_num, run=True, use_extra=False):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 43
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    def calcu_cos(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b_norm, a_norm[..., None]).flatten()

    if run:
        print("Initializing Evaluation...")
        img_to_text_dict = json.load(open(TEXT_FILE, "r"))

        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        processor_sat = AutoImageProcessor.from_pretrained(
            MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        test_bbox_dict = json.load(open(TEST_BBOX_FILE, "r"))

        model = Encoder_heading(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
        # model = Encoder_abla(
        #     model_name=MODEL_NAME, proj_dim=PROJECTION_DIM, usesg=True, useap=False
        # ).to(
        #     DEVICE
        # )

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
                heading_filename = f"{HEADING_FOLDER}/{index}_range250_heading0.png"
                search_path = f"{IMAGE_FOLDER}/{index}.png"
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
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        res_search = {}
        res_fused_query = {}
        res_heading = {}
        res_name = {}

        print(f"Starting batched inference on {len(dataset)} items...")

        with torch.inference_mode(), torch.amp.autocast("cuda"):
            for batch in tqdm(loader):
                if batch is None:
                    continue

                names = batch["name"]
                headings_batch = batch["heading"]
                query_inputs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
                search_inputs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
                input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)

                (
                    pred_anchor,
                    pred_heading,
                    text_feats,
                    anchor_feats,
                    grid_feats,
                    fused_feats,
                ) = model(query_inputs, search_inputs, input_ids)

                fused_query_feats = text_feats + anchor_feats

                grid_feats_np = grid_feats.float().cpu().numpy()
                fused_feats_np = fused_feats.float().cpu().numpy()
                pred_heading_np = pred_heading.float().cpu().numpy()

                for i, name in enumerate(names):
                    key = names[i]
                    res_search[key] = grid_feats_np[i]
                    res_fused_query[key] = fused_feats_np[i]

        SATELLITE_FOLDER = "/data/feihong/asian_univ"
        satellite_files = sorted(
            [f for f in os.listdir(SATELLITE_FOLDER) if f.endswith(".png")]
        )
        print(
            f"Processing {len(satellite_files)} satellite images from {SATELLITE_FOLDER}..."
        )
        
        if use_extra:
            satellite_batch = []
            satellite_names = []
            for sat_file in tqdm(satellite_files):
                sat_name = sat_file.replace(".png", "")
                sat_path = os.path.join(SATELLITE_FOLDER, sat_file)
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

            for i in range(0, len(satellite_batch), BATCH_SIZE):
                batch_pixels = torch.stack(satellite_batch[i : i + BATCH_SIZE]).to(DEVICE)
                batch_names = satellite_names[i : i + BATCH_SIZE]

                with torch.inference_mode(), torch.amp.autocast("cuda"):
                    grid_feats = model.ref_forward(batch_pixels)

                grid_feats_np = grid_feats.float().cpu().numpy()
                for j, name in enumerate(batch_names):
                    res_search[name] = grid_feats_np[j]

        np.savez("eval_search_heading.npz", **res_search)
        np.savez("eval_fused_query_heading.npz", **res_fused_query)
    #     np.savez("eval_pred_heading.npz", **res_heading)
    #     with open("eval_name_heading.json", "w") as f:
    #         json.dump(res_name, f)
    #     print("Evaluation feature extraction complete.")
    else:
        search_res = np.load("eval_search_heading.npz")
        fused_query_res = np.load("eval_fused_query_heading.npz")


        distances = json.load(open("/data/feihong/ckpt/distances.json", "r"))
        test_num = 100

        heading_correct = {0: 0, 90: 0, 180: 0, 270: 0}
        heading_total = {0: 0, 90: 0, 180: 0, 270: 0}

        # eval_keys = list(search_res.keys())
        # unique_names = list(set([k.split("_")[0] for k in eval_keys if "new" not in k]))
        eval_keys = [str(k) for k in search_res.keys()]
        unique_names = [k for k in eval_keys if "new" not in k]
        res = {}
        top1 = 0
        top5 = 0
        top10 = 0

        for key in tqdm(unique_names):
            fused_query_feature = fused_query_res[key]

            ex_img_list = random.sample(
                unique_names, min(test_num - 1, len(unique_names) - 1)
            )
            if key in ex_img_list:
                ex_img_list.remove(key)
            ex_img_list.append(key)

            candidate_indices = [99]
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
                if len(search_res[img_name].shape)==1:
                    grid_fea = search_res[img_name][None]
                else:
                    grid_fea = search_res[img_name]
                cos_sim_grid = calcu_cos(fused_query_feature, grid_fea)
                res[key].append(cos_sim_grid)

            img_res = np.array(res[key]).max(1).argsort()[-15:][::-1]

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


if __name__ == "__main__":
    exp_name = "unified_siglip_heading"
    end_num = 0.9
    exp_name = f"supp_{end_num}"
    save_dir = f"/data/feihong/ckpt/{exp_name}"

    if os.path.exists(save_dir):
        print(f"Experiment directory '{save_dir}' already exists.")
        print("Re-using directory. Logs and checkpoints might be overwritten.")
    else:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created experiment directory: {save_dir}")

    main(save_dir, end_num)
    eval(end_num,True)
    eval(end_num,False)
