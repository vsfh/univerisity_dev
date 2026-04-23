#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file train_sample.py
@desc Training SampleGeoLite for cross-view retrieval with InfoNCE loss.
     Removes bbox prediction branch, keeps only retrieval loss.
"""

import time
import random
import json
from pathlib import Path
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import argparse
import timm
import torch.nn.functional as F

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

IMG_SIZE = (768, 432)  # (width, height)
BATCH_SIZE = 8
NUM_EPOCHS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 50

UNIV_IMAGE_FOLDER = "/data/feihong/image_1024"
HEADING_FOLDER = "/data/feihong/range_250"
TRAIN_BBOX_FILE = "/data/feihong/univerisity_dev/runs/train.json"
TEST_BBOX_FILE = "/data/feihong/univerisity_dev/runs/test.json"
TRAIN_FILE = "/data/feihong/ckpt/train.txt"
TEST_FILE = "/data/feihong/ckpt/test.txt"
# UNIV_CROP_SIZE = (640, 640)
UNIV_DRONE_SIZE = (256, 256)
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

CVOGL_TRANSFORM = Compose(
    [
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

EMB_SIZE = 1024
PROJECTION_DIM = 1024
DATA_DIR = "/data/feihong"
DISTANCES_FILE = os.path.join(DATA_DIR, "ckpt/distances.json")


class TransformProcessorWrapper:
    def __init__(self, transform, image_size=IMG_SIZE):
        self.transform = transform
        if isinstance(image_size, (tuple, list)):
            self.image_size = (int(image_size[0]), int(image_size[1]))
        else:
            self.image_size = (int(image_size), int(image_size))
        self.size = {"height": self.image_size[1], "width": self.image_size[0]}

    def __call__(self, images, return_tensors="pt"):
        if isinstance(images, Image.Image):
            image_np = np.array(images)
        else:
            image_np = images
        pixel_values = self.transform(image_np)
        return {"pixel_values": pixel_values.unsqueeze(0)}


class DummyTokenizer:
    def __call__(
        self,
        text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ):
        del text, truncation, return_tensors
        token = torch.zeros((1, max_length), dtype=torch.long)
        return {"input_ids": token}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SampleGeoLite(nn.Module):
    """SampleGeoLite for retrieval (modified for InfoNCE loss)."""

    def __init__(self, emb_size=1024):
        super().__init__()

        model_name = "convnext_base.fb_in22k_ft_in1k_384"
        base_model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.query_model = base_model
        self.reference_model = base_model

        self.cross_attention = None

        self.query_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.reference_pool = nn.AdaptiveAvgPool2d((3, 3))

        self.query_projector = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),
            nn.ReLU(),
            nn.Linear(emb_size * 2, PROJECTION_DIM),
        )
        self.reference_projector = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),
            nn.ReLU(),
            nn.Linear(emb_size * 2, PROJECTION_DIM),
        )

    def retrieval_forward(self, query_imgs, reference_imgs):
        """Forward pass for retrieval with InfoNCE loss.

        Returns:
            query_feats: [B, PROJECTION_DIM] - pooled query features
            reference_feats: [B, 81, PROJECTION_DIM] - pooled grid features
        """
        query_fvisu = self.query_model.forward_features(query_imgs)
        reference_fvisu = self.reference_model.forward_features(reference_imgs)

        query_pooled = self.query_pool(query_fvisu).flatten(1)
        reference_grid = self.reference_pool(reference_fvisu)

        B, D, H, W = reference_grid.shape
        reference_flat = reference_grid.permute(0, 2, 3, 1).reshape(B, H * W, D)

        return query_pooled, reference_flat

    def forward(self, query_imgs, reference_imgs):
        return self.retrieval_forward(query_imgs, reference_imgs)


def info_nce_loss(query_feats, candidate_feats, positive_indices, temperature=0.07):
    """InfoNCE contrastive loss."""
    query_feats = nn.functional.normalize(query_feats, p=2, dim=1)
    candidate_feats = nn.functional.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature
    return nn.functional.cross_entropy(sim_matrix, positive_indices)


def train_epoch(
    loader,
    model,
    optimizer,
    epoch,
    scaler,
    use_amp=False,
    print_freq=PRINT_FREQ,
):
    """Train for one epoch."""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for batch_idx, batch in enumerate(loader):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        reference_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        local_indices = batch["index"].to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            query_feats, reference_feats = model(query_imgs, reference_imgs)

            B = query_feats.shape[0]
            candidate_feats = reference_feats.reshape(-1, PROJECTION_DIM)

            positive_indices = torch.zeros(B, B * 9, device=DEVICE)
            batch_offsets = torch.arange(B, device=DEVICE) * 9
            row_indices_broad = torch.arange(B, device=DEVICE).unsqueeze(1)
            col_offsets = torch.arange(9, device=DEVICE).unsqueeze(0)
            same_image_cols = batch_offsets.unsqueeze(1) + col_offsets

            positive_indices[row_indices_broad, same_image_cols] = 0.5
            global_positive_indices = local_indices + batch_offsets
            row_indices_flat = torch.arange(B, device=DEVICE)
            positive_indices[row_indices_flat, global_positive_indices] = 0.95

            loss = info_nce_loss(query_feats, candidate_feats, positive_indices)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), query_imgs.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0:
            print(
                f"Epoch: [{epoch}][{batch_idx}/{len(loader)}]\t"
                f"Time: {batch_time.val:.3f}\t"
                f"Loss: {losses.val:.4f} ({losses.avg:.4f})"
            )

    return losses.avg


def validate(loader, model, epoch, use_amp=False):
    """Validate model with InfoNCE loss."""
    model.eval()

    total_loss = 0
    total_samples = 0

    for batch in tqdm(loader, desc="Validating"):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        reference_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        local_indices = batch["index"].to(DEVICE)

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            query_feats, reference_feats = model(query_imgs, reference_imgs)

        B = query_feats.shape[0]
        candidate_feats = reference_feats.reshape(-1, PROJECTION_DIM)

        positive_indices = torch.zeros(B, B * 9, device=DEVICE)
        batch_offsets = torch.arange(B, device=DEVICE) * 9
        row_indices_broad = torch.arange(B, device=DEVICE).unsqueeze(1)
        col_offsets = torch.arange(9, device=DEVICE).unsqueeze(0)
        same_image_cols = batch_offsets.unsqueeze(1) + col_offsets

        positive_indices[row_indices_broad, same_image_cols] = 0.5
        global_positive_indices = local_indices + batch_offsets
        row_indices_flat = torch.arange(B, device=DEVICE)
        positive_indices[row_indices_flat, global_positive_indices] = 0.95

        loss = info_nce_loss(query_feats, candidate_feats, positive_indices)

        total_loss += loss.item() * B
        total_samples += B

    return total_loss / total_samples if total_samples > 0 else 0


def load_distances():
    """Load distances.json and strip .kml extensions from keys."""
    import json

    with open(DISTANCES_FILE, "r") as f:
        distances = json.load(f)

    cleaned = {}
    for kml1, inner_dict in distances.items():
        key1 = kml1.replace(".kml", "")
        cleaned[key1] = {}
        for kml2, dist in inner_dict.items():
            key2 = kml2.replace(".kml", "")
            cleaned[key1][key2] = dist

    return cleaned


def calcu_cos(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm).flatten()


def eval_model(run=False, checkpoint_path=None):
    use_amp = torch.cuda.is_available()
    if run:
        print("Loading test dataset...")
        processor = TransformProcessorWrapper(CVOGL_TRANSFORM, args.img_size)
        tokenizer = DummyTokenizer()
        test_dataset = ShiftedSatelliteDroneDataset(
            processor=processor,
            processor_sat=processor,
            tokenizer=tokenizer,
            split="test",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        model = SampleGeoLite()
        model = model.cuda()

        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print("Extracting features from test set...")
        model.eval()

        res_satellite = {}
        res_drone = {}

        for batch in tqdm(test_loader):
            drone = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
            satellite = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
            labels = [Path(path).stem for path in batch["satellite_path"]]

            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                feat1, feat2 = model(drone, satellite)
                feat1 = F.normalize(feat1, p=2, dim=1)
                feat2 = F.normalize(feat2, p=2, dim=1)

            for i in range(satellite.size(0)):
                sample_name = labels[i]
                res_satellite[sample_name] = feat1[i].cpu().numpy()
                res_drone[sample_name] = feat2[i].cpu().numpy()

        SATELLITE_FOLDER = "/data/feihong/asian_univ"
        satellite_files = sorted(
            [f for f in os.listdir(SATELLITE_FOLDER) if f.endswith(".png")]
        )
        print(
            f"Processing {len(satellite_files)} satellite images from {SATELLITE_FOLDER}..."
        )

        for sat_file in tqdm(satellite_files):
            sat_name = sat_file.replace(".png", "")
            sat_path = os.path.join(SATELLITE_FOLDER, sat_file)
            try:
                sat_image = Image.open(sat_path).convert("RGB")
                sat_image = sat_image.crop((840, 0, 3000, 2160)).resize(args.img_size)
                sat_img_np = np.array(sat_image)
                sat_img = CVOGL_TRANSFORM(sat_img_np).unsqueeze(0).to(DEVICE)

                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    feat, _ = model(sat_img, sat_img)
                    feat = F.normalize(feat, p=2, dim=1)

                res_satellite[sat_name] = feat[0].cpu().numpy()
            except Exception as e:
                print(f"Error processing {sat_file}: {e}")
                continue

        np.savez("eval_search_fsra.npz", **res_satellite)
        np.savez("eval_drone_fsra.npz", **res_drone)
        print("Evaluation feature extraction complete.")
    else:
        drone_res = np.load(
            "/data/feihong/univerisity_dev/grounding_cvos/eval_search_fsra.npz"
        )
        search_res = np.load(
            "/data/feihong/univerisity_dev/grounding_cvos/eval_drone_fsra.npz"
        )

        distances = load_distances()
        test_num = 100
        test_list = [k for k in search_res.keys() if not 'new' in k]
        res = {}
        top1 = 0
        top5 = 0
        top10 = 0

        if not test_list:
            print("No evaluation data found in npz files.")
            return

        for key in tqdm(test_list):
            drone_feature = drone_res[key]

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
                cos_sim = calcu_cos(drone_feature, search_res[img_name])
                res[key].append(cos_sim)

            img_res = np.array(res[key]).mean(1).argsort()[-15:][::-1]

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


def main(args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_name = args.savename if args.savename else "sample_retrieval_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Creating datasets from shared data source...")
    processor = TransformProcessorWrapper(CVOGL_TRANSFORM, args.img_size)
    tokenizer = DummyTokenizer()

    train_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor,
        tokenizer=tokenizer,
        split="train",
    )
    val_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor,
        tokenizer=tokenizer,
        split="test",
    )

    print(f"Found {len(train_dataset)} training pairs, {len(val_dataset)} validation pairs")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    print("Creating model...")
    model = SampleGeoLite(emb_size=EMB_SIZE).to(DEVICE)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    print(f"Starting training for {args.max_epoch} epochs...")
    for epoch in range(args.max_epoch):
        train_loss = train_epoch(
            train_loader,
            model,
            optimizer,
            epoch,
            scaler=scaler,
            use_amp=use_amp,
            print_freq=args.print_freq,
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        # writer.add_scalar("Metrics/top1_acc", top1, epoch)
        # writer.add_scalar("Metrics/top5_acc", top5, epoch)
        # writer.add_scalar("Metrics/top10_acc", top10, epoch)

        print(
            f"Epoch {epoch + 1}/{args.max_epoch}:\t"
            f"Train Loss: {train_loss:.4f}\t"
            # f"Top1: {top1:.4f}\t"
            # f"Top5: {top5:.4f}\t"
            # f"Top10: {top10:.4f}"
        )
        os.makedirs(args.checkpoint, exist_ok=True)
        torch.save(model.state_dict(), f"{args.checkpoint}/last.pth")

    print("\nTraining complete. Saved checkpoint to last.pth")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SampleGeoLite Retrieval Training")
    parser.add_argument("--gpu", default="1", help="GPU id")
    parser.add_argument(
        "--num-workers", type=int, default=8, help="num workers for data loading"
    )
    parser.add_argument(
        "--max-epoch", type=int, default=NUM_EPOCHS, help="training epoch"
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=list(IMG_SIZE),
        help="image size as WIDTH HEIGHT",
    )
    parser.add_argument(
        "--savename",
        type=str,
        default="sample_retrieval",
        help="Name head for saved model",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="random seed")
    parser.add_argument(
        "--print-freq", type=int, default=PRINT_FREQ, help="print frequency"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/feihong/ckpt/retrieval_sample4geo",
        help="Path to save model checkpoints",
    )
    args = parser.parse_args()
    args.img_size = (int(args.img_size[0]), int(args.img_size[1]))

    os.makedirs(args.checkpoint, exist_ok=True)

    main(args)
    # eval_model(run=True, checkpoint_path="/data/feihong/ckpt/retrieval_sample4geo/best.pth")
    # eval_model(run=False)
