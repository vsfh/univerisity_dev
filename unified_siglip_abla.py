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

from model import Encoder_abla, Encoder_heading
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

NUM_EPOCHS = 25
BATCH_SIZE = 15
GRAD_ACCUMULATION_STEPS = 3
LEARNING_RATE = 1e-4
LR_MIN = 1e-10
COSINE_EPOCHS = 25
BBOX_LOSS_WEIGHT = 0.1
HEADING_LOSS_WEIGHT = .1
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


def get_loss_weights(epoch, num_epochs):
    progress = epoch / num_epochs
    bbox_weight = 0.5 * (1 + np.cos(np.pi * progress)) / 2
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


class HeadingDataset(Dataset):
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

        # if self.mode == "train":
        #     heading = random.choice([0, 90, 180, 270])
        #     query_path = query_path.replace("heading0", f"heading{heading}")
        # else:
        #     heading = 0
        if 'heading' in query_path:
            name = os.path.basename(query_path).split("_")[0]
            heading = random.choice([0, 90, 180, 270])
            query_path = query_path.replace("heading0", f"heading{heading}")
        else:
            name = query_path.split('/')[-2]
            heading = 0

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

        if 'heading' in query_path:
            query_image = query_image.crop((420, 0, 1500, 1080))
        # score = random.random()
        # if score > 0.5:
        #     crop_length = int(150 * (score - 0.5))
        #     query_image = query_image.crop(
        #         (crop_length, crop_length, 1080 - crop_length, 1080 - crop_length)
        #     )

        query_image = resize_drone_image(query_image, DRONE_SIZE)

        bbox_key = name
        bbox = None
        if self.bbox_dict:
            bbox = self.bbox_dict.get(bbox_key, None)
        if bbox is None:
            bbox = (1536, 656, 2268, 1374)
            print('error')

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

            img_text_loss = info_nce_loss(fused_feats, candidate_feats, positive_indices)
        else:

            query_feats = F.normalize(fused_feats, p=2, dim=1)
            candidate_feats = F.normalize(grid_feats, p=2, dim=1)
                    
            # Similarity matrix: (B, B)
            sim_matrix = torch.matmul(query_feats, candidate_feats.T)                
            # Positive indices: diagonal (each query's positive is itself)
            positive_indices = torch.arange(B, device=query_feats.device)
                    
            img_text_loss = F.cross_entropy(sim_matrix, positive_indices)
        pred_anchor = pred_anchor.view(
            B, 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )
        for i in range(B):
            _, best_anchor_gi_gj = build_target(
                ori_gt_bbox[i:i+1], anchors_full, img_size, pred_anchor.shape[3]
            )

            accu_list, accu_center, iou, _, _, _ = eval_iou_acc(
                pred_anchor[i:i+1],
                ori_gt_bbox[i:i+1],
                anchors_full,
                best_anchor_gi_gj[:, 1],
                best_anchor_gi_gj[:, 2],
                img_size,
                iou_threshold_list=[0.5, 0.25],
            )



            heading_loss = F.mse_loss(pred_heading[i:i+1], heading_target[i:i+1])

            accu50_meter.update(accu_list[0].item(), 1)
            accu25_meter.update(accu_list[1].item(), 1)
            iou_meter.update(iou.item(), 1)
            heading_meter.update(heading_loss.item(), 1)
        info_nce_meter.update(img_text_loss.item(), query_imgs.shape[0])

    return (
        accu50_meter.avg,
        accu25_meter.avg,
        iou_meter.avg,
        info_nce_meter.avg,
        heading_meter.avg,
    )


def main(save_path):
    usesg = True
    useap = True
    usegf = True
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        kwargs_handlers=[ddp_kwargs],
    )

    not_update = 0
    soft_label = 0.5
    exp_name = save_path.split("/")[-1] if save_path else "default_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    train_image_pairs = []
    test_image_pairs = []
    print("Reading train/test IDs from files...")
    train_ids = set()
    with open("/data/feihong/ckpt/train.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            query_path = f"{HEADING_FOLDER}/{name}_range250_heading0.png"
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            train_image_pairs.append((query_path, search_path))

    test_ids = set()
    with open("/data/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            query_path = f"{HEADING_FOLDER}/{name}_range250_heading0.png"
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            test_image_pairs.append((query_path, search_path))

    print(f"Train IDs: {len(train_image_pairs)}, Test IDs: {len(test_image_pairs)}")

    print("Building heading image pairs (heading 0 only)...")


    # for img_id in train_ids:
    #     heading_filename = f"{HEADING_FOLDER}/{img_id}_range250_heading0.png"
    #     search_path = f"{IMAGE_FOLDER}/{img_id}.png"
    #     if os.path.exists(heading_filename) and os.path.exists(search_path):
    #         train_image_pairs.append((heading_filename, search_path))

    # for img_id in test_ids:
    #     heading_filename = f"{HEADING_FOLDER}/{img_id}_range250_heading0.png"
    #     search_path = f"{IMAGE_FOLDER}/{img_id}.png"
    #     if os.path.exists(heading_filename) and os.path.exists(search_path):
    #         test_image_pairs.append((heading_filename, search_path))

    print(f"Train: {len(train_image_pairs)}, Test: {len(test_image_pairs)}")

    print("Loading models and processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor_sat = AutoImageProcessor.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = Encoder_abla(
        model_name=MODEL_NAME, proj_dim=PROJECTION_DIM, usesg=usesg, useap=useap
    )

    img_to_text_dict = json.load(open(TEXT_FILE, "r"))
    train_bbox_dict = json.load(open(TRAIN_BBOX_FILE, "r"))
    test_bbox_dict = json.load(open(TEST_BBOX_FILE, "r"))

    anchors_full = get_tensor_anchors(accelerator.device)

    print("Setting up dataset and dataloader...")
    train_dataset = HeadingDataset(
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

    test_dataset = HeadingDataset(
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

    BBOX_LR = 5e-5
    RETRIEVAL_LR = 1e-5
    # optimizer = torch.optim.AdamW(
    #     [
    #         {"params": bbox_params, "lr": BBOX_LR},
    #         {"params": retrieval_params, "lr": RETRIEVAL_LR},
    #     ]
    # )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

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

                if useap:
                    candidate_feats = grid_feats.reshape(-1, PROJECTION_DIM)
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
                    if usegf:
                        img_text_loss = 0.7 * info_nce_loss(
                            fused_feats, candidate_feats, positive_indices
                        )
                        img_text_loss += 0.3 * info_nce_loss(
                            anchor_feats, candidate_feats, positive_indices
                        )
                    else:
                        img_text_loss = info_nce_loss(
                            (anchor_feats+text_feats)/2, candidate_feats, positive_indices
                        )    
                else:
                    # Both shapes: (B, 768)
                    query_feats = F.normalize(fused_feats, p=2, dim=1)
                    candidate_feats = F.normalize(grid_feats, p=2, dim=1)
                    
                    # Similarity matrix: (B, B)
                    sim_matrix = torch.matmul(query_feats, candidate_feats.T)
                    
                    # Positive indices: diagonal (each query's positive is itself)
                    positive_indices = torch.arange(B, device=query_feats.device)
                    
                    img_text_loss = F.cross_entropy(sim_matrix, positive_indices)

                heading_loss = F.mse_loss(pred_heading, heading_target)

                bbox_weight, retrieval_weight = get_loss_weights(epoch, COSINE_EPOCHS)
                loss = (
                    retrieval_weight * img_text_loss
                    + bbox_weight * bbox_loss
                    + heading_loss * HEADING_LOSS_WEIGHT
                )
                # loss = img_text_loss + bbox_loss + heading_loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()
            total_heading_loss += heading_loss.item()
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "bbox_loss": f"{bbox_loss.item():.4f}",
                    "heading_loss": f"{heading_loss.item():.4f}",
                    # "img_text_loss": f"{img_text_loss.item():.4f}",
                }
            )
            writer.add_scalar(
                "Loss/train_batch", loss.item(), epoch * len(train_dataloader) + i
            )

        avg_loss = total_loss / len(train_dataloader)
        avg_bbox_loss = total_bbox_loss / len(train_dataloader)
        avg_heading_loss = total_heading_loss / len(train_dataloader)

        model.eval()
        accu50, accu25, val_iou, val_info_nce, val_heading = validation(
            test_dataloader,
            model,
            accelerator,
            anchors_full,
            UNIV_SAT_SIZE[0],
            useap=useap
        )
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Metrics/val_iou", val_iou, epoch)
        writer.add_scalar("Metrics/val_accu50", accu50, epoch)
        writer.add_scalar("Metrics/val_accu25", accu25, epoch)
        writer.add_scalar("Metrics/val_heading", val_heading, epoch)
        accelerator.print(
            f"Epoch {epoch + 1} finished. "
            f"Avg train Loss: {avg_loss:.4f} (bbox: {avg_bbox_loss:.4f}, heading: {avg_heading_loss:.4f}). "
            f"IoU: {val_iou:.4f}, Accu50: {accu50:.4f}, Accu25: {accu25:.4f}, "
            f"InfoNCE: {val_info_nce:.4f}, Heading: {val_heading:.4f}"
        )

        if accelerator.is_main_process:
            # if val_info_nce < min_info:
            #     os.makedirs(save_path, exist_ok=True)
            #     unwrapped_model = accelerator.unwrap_model(model)

            #     save_filename = f"{save_path}/best_info_{not_update}.pth"
            #     accelerator.save(unwrapped_model.state_dict(), save_filename)

            #     print(f"Saved checkpoint info: {save_filename}")
            #     min_info = val_info_nce
            #     not_update += 1
            if max_iou < val_iou:
                os.makedirs(save_path, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)

                save_filename = f"{save_path}/best_iou.pth"
                accelerator.save(unwrapped_model.state_dict(), save_filename)

                print(f"Saved checkpoint iou: {save_filename}")
                max_iou = val_iou
                not_update += 1
        scheduler.step()

    accelerator.print("Training complete.")
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


def eval(run=False):
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

    print("Loading model and processors...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor_sat = AutoImageProcessor.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = Encoder_abla(
        model_name=MODEL_NAME, proj_dim=PROJECTION_DIM, useap=True, usesg=True
    ).to(DEVICE)

    model_path = "/data/feihong/ckpt/unified_siglip_heading/best_info_42.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded checkpoint: {model_path}")
    else:
        print(f"Checkpoint not found: {model_path}")
        return
    model.eval()

    print("Building eval pairs with all heading directions...")
    img_to_text_dict = json.load(open(TEXT_FILE, "r"))

    headings = [0]

    eval_pairs = []
    test_ids = set()
    with open("/data/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            img_id = line.strip().split("/")[-2]
            test_ids.add(img_id)
            for heading in headings:
                heading_filename = (
                    f"{HEADING_FOLDER}/{img_id}_range250_heading{heading}.png"
                )
                search_path = f"{IMAGE_FOLDER}/{img_id}.png"
                if os.path.exists(heading_filename) and os.path.exists(search_path):
                    eval_pairs.append((heading_filename, search_path, heading))

    print(f"Total eval pairs: {len(eval_pairs)}")

    heading_correct = {0: 0, 90: 0, 180: 0, 270: 0}
    heading_total = {0: 0, 90: 0, 180: 0, 270: 0}

    all_pred_headings = []
    all_target_headings = []

    res_search = {}
    res_fused_query = {}

    print("Running inference...")
    batch_size = BATCH_SIZE

    for i in tqdm(range(0, len(eval_pairs), batch_size)):
        batch_pairs = eval_pairs[i : i + batch_size]
        batch_queries = []
        batch_searches = []
        batch_headings = []
        batch_names = []
        batch_texts = []

        for query_path, search_path, heading in batch_pairs:
            try:
                query_image = Image.open(query_path).convert("RGB")
                search_image = Image.open(search_path).convert("RGB")

                query_image = query_image.crop((420, 0, 1500, 1080))
                query_image = resize_drone_image(query_image, DRONE_SIZE)

                search_image = search_image.crop((840, 0, 3000, 2160))
                search_image = search_image.resize(UNIV_SAT_SIZE)

                name = os.path.basename(query_path).split("_")[0]
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
                    "Description",
                    "()",
                ]:
                    text_description = text_description.replace(noun, "")

                query_pixels = processor(images=query_image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                search_pixels = processor_sat(images=search_image, return_tensors="pt")[
                    "pixel_values"
                ][0]

                batch_queries.append(query_pixels)
                batch_searches.append(search_pixels)
                batch_headings.append(heading)
                batch_names.append(name)
                batch_texts.append(text_description)
            except Exception as e:
                print(f"Error loading {query_path}: {e}")
                continue

        if len(batch_queries) == 0:
            continue

        query_tensor = torch.stack(batch_queries).to(DEVICE)
        search_tensor = torch.stack(batch_searches).to(DEVICE)

        text_inputs = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(DEVICE)

        with torch.inference_mode(), torch.amp.autocast("cuda"):
            (
                pred_anchor,
                pred_heading,
                text_feats,
                anchor_feats,
                grid_feats,
                fused_feats,
            ) = model(query_tensor, search_tensor, input_ids)

        pred_heading_np = pred_heading.float().cpu().numpy()
        grid_feats_np = grid_feats.float().cpu().numpy()
        fused_feats_np = anchor_feats.float().cpu().numpy()

        for j, heading in enumerate(batch_headings):
            pred = pred_heading_np[j]
            target = HEADING_TO_TARGET[heading]

            all_pred_headings.append(pred)
            all_target_headings.append(target)

            pred_heading_val = pred_to_heading(pred)

            heading_total[heading] += 1
            if pred_heading_val == heading:
                heading_correct[heading] += 1

            name = batch_names[j]
            if heading == 0:
                res_search[name] = grid_feats_np[j]
                res_fused_query[name] = fused_feats_np[j]

    print("\n=== Heading Prediction Results ===")
    for h in headings:
        if heading_total[h] > 0:
            acc = heading_correct[h] / heading_total[h] * 100
            print(
                f"Heading {h:3d}: {heading_correct[h]:4d} / {heading_total[h]:4d} ({acc:.2f}%)"
            )

    total_correct = sum(heading_correct.values())
    total_samples = sum(heading_total.values())
    overall_acc = total_correct / total_samples * 100
    print(f"Overall:    {total_correct:4d} / {total_samples:4d} ({overall_acc:.2f}%)")

    pred_headings_tensor = torch.tensor(all_pred_headings, dtype=torch.float32)
    target_headings_tensor = torch.tensor(all_target_headings, dtype=torch.float32)
    mean_mse = F.mse_loss(pred_headings_tensor, target_headings_tensor).item()
    print(f"\nMean MSE of Heading: {mean_mse:.6f}")
    print(f"Heading Accuracy: {overall_acc:.2f}%")

    np.savez("eval_search_siglip.npz", **res_search)
    np.savez("eval_fused_query_siglip.npz", **res_fused_query)
    print("\nFeature extraction complete.")

    SATELLITE_FOLDER = "/data/feihong/asian_univ"
    satellite_files = sorted(
        [f for f in os.listdir(SATELLITE_FOLDER) if f.endswith(".png")]
    )
    print(
        f"Processing {len(satellite_files)} satellite images from {SATELLITE_FOLDER}..."
    )

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

    np.savez("eval_search_siglip.npz", **res_search)
    print("Satellite feature extraction complete.")

    search_res = np.load("eval_search_siglip.npz")
    fused_query_res = np.load("eval_fused_query_siglip.npz")

    distances = json.load(open("/data/feihong/ckpt/distances.json", "r"))
    test_num = 100
    test_list = [k for k in search_res.keys() if not "new" in k]

    if not test_list:
        print("No evaluation data found in npz files.")
        return

    res = {}
    top1 = 0
    top5 = 0
    top10 = 0

    for key in tqdm(test_list):
        fused_query_feature = fused_query_res[key]

        ex_img_list = random.sample(test_list, min(test_num - 1, len(test_list) - 1))
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

    print(f"\n=== Retrieval Results ===")
    print(f"Top 1: {top1} / {len(test_list)} ({top1 / len(test_list) * 100:.2f}%)")
    print(f"Top 5: {top5} / {len(test_list)} ({top5 / len(test_list) * 100:.2f}%)")
    print(f"Top 10: {top10} / {len(test_list)} ({top10 / len(test_list) * 100:.2f}%)")


def eval_old(run=False):
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
            if "heading" in query_path:
                name = query_path.split("/")[-1].split("_")[0]
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
            # Apply crop for heading0 images before resize
            if "_range250_heading0" in query_path:
                query_image = query_image.crop((420, 0, 1500, 1080))
                crop_length = 150
                query_image = query_image.crop(
                    (crop_length, crop_length, 1080 - crop_length, 1080 - crop_length)
                )
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
        encoder = Encoder_abla(
            model_name=MODEL_NAME, proj_dim=PROJECTION_DIM, useap=False, usesg=False
        ).to(DEVICE)
        # Using compile for speedup
        # encoder = torch.compile(encoder)

        model_path = f"../ckpt/unified_siglip_heading/best_info_52.pth"
        if os.path.exists(model_path):
            encoder.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded checkpoint: {model_path}")
        else:
            print("error")
            return

        encoder.eval()

        # Prepare Data List
        eval_list = []
        with open("/data/feihong/ckpt/test.txt", "r") as f:
            for line in f:
                img_path = line.strip().replace("01.", "41.")
                index = line.strip().split("/")[-2]
                img_path = f"/data/feihong/range_250/{index}_range250_heading0.png"
                eval_list.append(img_path)

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
                _, _, _, anchor_feats, grid_feats, fused_query_feats = encoder.forward(
                    query_inputs, search_inputs, input_ids
                )

                # Move to CPU and store
                grid_feats_np = grid_feats.float().cpu().numpy()
                fused_feats_np = fused_query_feats.float().cpu().numpy()

                for i, name in enumerate(names):
                    res_search[name] = grid_feats_np[i]
                    res_fused_query[name] = fused_feats_np[i]

        SATELLITE_FOLDER = "/data/feihong/asian_univ"
        satellite_files = sorted(
            [f for f in os.listdir(SATELLITE_FOLDER) if f.endswith(".png")]
        )
        print(
            f"Processing {len(satellite_files)} satellite images from {SATELLITE_FOLDER}..."
        )

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
                grid_feats = encoder.ref_forward(batch_pixels)

            grid_feats_np = grid_feats.float().cpu().numpy()
            for j, name in enumerate(batch_names):
                res_search[name] = grid_feats_np[j]

        np.savez("eval_search_clip.npz", **res_search)
        np.savez("eval_fused_query_clip.npz", **res_fused_query)
        print("Evaluation feature extraction complete.")
    else:
        search_res = np.load("eval_search_clip.npz")
        fused_query_res = np.load("eval_fused_query_clip.npz")

        distances = json.load(open("/data/feihong/ckpt/distances.json", "r"))
        test_num = 100
        test_list = [k for k in search_res.keys() if not "new" in k]
        # test_list = [k for k in search_res.keys()]
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


def eval_heading():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 43
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("Loading model and processors...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor_sat = AutoImageProcessor.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = Encoder_abla(
        model_name=MODEL_NAME, proj_dim=PROJECTION_DIM, useap=False, usesg=False
    ).to(DEVICE)

    model_path = "/data/feihong/ckpt/unified_siglip_heading/best_iou_59.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded checkpoint: {model_path}")
    else:
        print(f"Checkpoint not found: {model_path}")
        return
    model.eval()

    print("Building eval pairs with all heading directions...")
    img_to_text_dict = json.load(open(TEXT_FILE, "r"))

    headings = [0, 90, 180, 270]
    heading_to_target = {
        0: [0.0, 0.0],
        90: [0.0, 1.0],
        180: [1.0, 0.0],
        270: [1.0, 1.0],
    }

    eval_pairs = []
    with open("/data/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            img_id = line.strip().split("/")[-2]
            for heading in headings:
                heading_filename = (
                    f"{HEADING_FOLDER}/{img_id}_range250_heading{heading}.png"
                )
                search_path = f"{IMAGE_FOLDER}/{img_id}.png"
                if os.path.exists(heading_filename) and os.path.exists(search_path):
                    eval_pairs.append((heading_filename, search_path, heading))

    print(f"Total eval pairs: {len(eval_pairs)}")

    heading_correct = {0: 0, 90: 0, 180: 0, 270: 0}
    heading_total = {0: 0, 90: 0, 180: 0, 270: 0}

    print("Running inference...")
    batch_size = BATCH_SIZE

    for i in tqdm(range(0, len(eval_pairs), batch_size)):
        batch_pairs = eval_pairs[i : i + batch_size]
        batch_queries = []
        batch_searches = []
        batch_headings = []

        for query_path, search_path, heading in batch_pairs:
            try:
                query_image = Image.open(query_path).convert("RGB")
                search_image = Image.open(search_path).convert("RGB")

                query_image = query_image.crop((420, 0, 1500, 1080))
                query_image = resize_drone_image(query_image, DRONE_SIZE)

                search_image = search_image.crop((840, 0, 3000, 2160))
                search_image = search_image.resize(UNIV_SAT_SIZE)

                query_pixels = processor(images=query_image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                search_pixels = processor_sat(images=search_image, return_tensors="pt")[
                    "pixel_values"
                ][0]

                batch_queries.append(query_pixels)
                batch_searches.append(search_pixels)
                batch_headings.append(heading)
            except Exception as e:
                print(f"Error loading {query_path}: {e}")
                continue

        if len(batch_queries) == 0:
            continue

        query_tensor = torch.stack(batch_queries).to(DEVICE)
        search_tensor = torch.stack(batch_searches).to(DEVICE)

        with torch.inference_mode(), torch.amp.autocast("cuda"):
            (
                pred_anchor,
                pred_heading,
                text_feats,
                anchor_feats,
                grid_feats,
                fused_feats,
            ) = model(query_tensor, search_tensor, None)

        pred_heading_np = pred_heading.float().cpu().numpy()

        for j, heading in enumerate(batch_headings):
            pred = pred_heading_np[j]
            pred_heading_val = pred_to_heading(pred)

            heading_total[heading] += 1
            if pred_heading_val == heading:
                heading_correct[heading] += 1

    print("\n=== Heading Prediction Results ===")
    for h in headings:
        if heading_total[h] > 0:
            acc = heading_correct[h] / heading_total[h] * 100
            print(
                f"Heading {h:3d}: {heading_correct[h]:4d} / {heading_total[h]:4d} ({acc:.2f}%)"
            )

    total_correct = sum(heading_correct.values())
    total_samples = sum(heading_total.values())
    overall_acc = total_correct / total_samples * 100
    print(f"Overall:    {total_correct:4d} / {total_samples:4d} ({overall_acc:.2f}%)")


if __name__ == "__main__":
    exp_name = "unified_siglip_abla_grounding"
    exp_name = "abla_wo_loss_weight"
    exp_name = "abla_wo_gated_fusion"
    exp_name = "abla_wo_adaptive_pool"
    exp_name = "abla_wo_stop_gardient"
    exp_name = "abla_all_good"
    save_dir = f"/data/feihong/ckpt/{exp_name}"

    if os.path.exists(save_dir):
        print(f"Experiment directory '{save_dir}' already exists.")
        print("Re-using directory. Logs and checkpoints might be overwritten.")
    else:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created experiment directory: {save_dir}")

    main(save_dir)
    # eval()
    # eval_old(True)
    eval_old(False)
    # eval_heading()
