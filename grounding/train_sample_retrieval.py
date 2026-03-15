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
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
from glob import glob
import argparse
import timm
import torch.nn.functional as F

from ground_cvos import (
    format_satellite_img_bbox,
    resize_drone_image,
)

SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

IMG_SIZE = 640
BATCH_SIZE = 4
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 50

UNIV_IMAGE_FOLDER = "/data/feihong/image_1024"
HEADING_FOLDER = "/data/feihong/range_250"
TRAIN_BBOX_FILE = "/data/feihong/univerisity_dev/runs/train.json"
TEST_BBOX_FILE = "/data/feihong/univerisity_dev/runs/test.json"
TRAIN_FILE = "/data/feihong/ckpt/train.txt"
TEST_FILE = "/data/feihong/ckpt/test.txt"
UNIV_CROP_SIZE = (640, 640)
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


def train_epoch(loader, model, optimizer, epoch, print_freq=PRINT_FREQ):
    """Train for one epoch."""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for batch_idx, batch in enumerate(loader):
        query_imgs = batch["query_imgs"].to(DEVICE)
        reference_imgs = batch["rs_imgs"].to(DEVICE)
        local_indices = batch["index"].to(DEVICE)

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

        optimizer.zero_grad()
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


def validate(loader, model, epoch):
    """Validate model with InfoNCE loss."""
    model.eval()

    total_loss = 0
    total_samples = 0

    for batch in tqdm(loader, desc="Validating"):
        query_imgs = batch["query_imgs"].to(DEVICE)
        reference_imgs = batch["rs_imgs"].to(DEVICE)
        local_indices = batch["index"].to(DEVICE)

        with torch.no_grad():
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
    use_gpu = torch.cuda.is_available()

    if run:
        print("Loading distances.json...")
        distances = load_distances()

        print("Loading train dataset to get correct number of classes...")
        val_pairs = []
        with open(TEST_FILE, "r") as f:
            for line in f:
                # query_path = line.strip().replace('01.','41.')
                # name = query_path.split("/")[-2]

                name = line.strip().split('/')[-2]
                query_path = f'/data/feihong/range_250/{name}_range250_heading0.png'
                search_path = f"{UNIV_IMAGE_FOLDER}/{name}.png"
                if os.path.exists(search_path):
                    val_pairs.append((query_path, search_path))
        bbox_dict = {}
        for f in [TRAIN_BBOX_FILE, TEST_BBOX_FILE]:
            with open(f, "r") as file:
                data = json.load(file)
                bbox_dict.update(data)
        test_dataset = TargetSearchDataset(
            image_pairs=val_pairs,
            bbox_dict=bbox_dict,
            mode="test",
            transform=CVOGL_TRANSFORM,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=args.num_workers,
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
            drone = batch["query_imgs"].to(DEVICE)
            satellite = batch["rs_imgs"].to(DEVICE)
            labels = batch["name"]
            # for satellite, drone, labels in tqdm(test_loader, desc="Extracting features"):
            if use_gpu:
                satellite = satellite.cuda()
                drone = drone.cuda()

            with torch.no_grad():
                feat1, feat2 = model(satellite, drone)
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
                sat_image = sat_image.crop((840, 0, 3000, 2160)).resize((640, 640))
                sat_img_np = np.array(sat_image)
                sat_img = CVOGL_TRANSFORM(sat_img_np).unsqueeze(0).cuda()

                with torch.no_grad():
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


class TargetSearchDataset(Dataset):
    """Dataset for retrieval training with InfoNCE loss."""

    def __init__(self, image_pairs, bbox_dict, mode="train", transform=None):
        self.image_paths = image_pairs
        self.bbox_dict = bbox_dict
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        query_path, search_path = self.image_paths[idx]

        img_id = query_path.split("/")[-2]

        # choice = []
        # for number in ["01.", "21.", "31.", "41.", "51."]:
        #     new_query_path = query_path.replace("01.", number)
        #     if not os.path.exists(new_query_path):
        #         continue
        #     choice.append(new_query_path)

        # heading0_path = f"{HEADING_FOLDER}/{img_id}_range250_heading0.png"
        # if os.path.exists(heading0_path):
        #     choice.append(heading0_path)

        # if choice:
        #     query_path = random.sample(choice, 1)[0]

        if "_range250_heading0" in query_path:
            name = query_path.split("/")[-1].split("_")[0]
        else:
            name = query_path.split("/")[-2]

        try:
            query_image = Image.open(query_path).convert("RGB")
            search_image = Image.open(search_path).convert("RGB")
        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self))

        if "_range250_heading0" in query_path:
            query_image = query_image.crop((420, 0, 1500, 1080))

        query_image = resize_drone_image(query_image, UNIV_DRONE_SIZE)

        bbox_key = name
        bbox = self.bbox_dict.get(bbox_key, None)
        if bbox is None:
            bbox = (1536, 656, 2268, 1374)

        search_image, normalized_bbox = format_satellite_img_bbox(
            search_image, bbox, mode=self.mode, target_size=UNIV_CROP_SIZE
        )

        center_x = (normalized_bbox[0] + normalized_bbox[2]) / 2
        center_y = (normalized_bbox[1] + normalized_bbox[3]) / 2

        col_idx = min(int(center_x * 3), 2)
        row_idx = min(int(center_y * 3), 2)
        index = row_idx * 3 + col_idx

        query_img_np = np.array(query_image)
        rs_img_np = np.array(search_image)

        if self.transform:
            query_img = self.transform(query_img_np)
            rs_img = self.transform(rs_img_np)
        else:
            query_img = (
                torch.tensor(query_img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
            )
            rs_img = (
                torch.tensor(rs_img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
            )

        return {
            "query_imgs": query_img,
            "rs_imgs": rs_img,
            "index": torch.tensor(index, dtype=torch.long),
            "name": name,
        }


def main(args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_name = args.savename if args.savename else "sample_retrieval_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Loading bbox annotations...")
    bbox_dict = {}
    for f in [TRAIN_BBOX_FILE, TEST_BBOX_FILE]:
        with open(f, "r") as file:
            data = json.load(file)
            bbox_dict.update(data)

    print(f"Loaded {len(bbox_dict)} bbox annotations")

    print("Loading image pairs...")
    train_pairs = []
    with open(TRAIN_FILE, "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{UNIV_IMAGE_FOLDER}/{name}.png"
            if os.path.exists(search_path):
                train_pairs.append((query_path, search_path))

    val_pairs = []
    with open(TEST_FILE, "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{UNIV_IMAGE_FOLDER}/{name}.png"
            if os.path.exists(search_path):
                val_pairs.append((query_path, search_path))

    print(f"Found {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs")

    print("Creating datasets...")
    train_dataset = TargetSearchDataset(
        image_pairs=train_pairs,
        bbox_dict=bbox_dict,
        mode="train",
        transform=CVOGL_TRANSFORM,
    )
    val_dataset = TargetSearchDataset(
        image_pairs=val_pairs,
        bbox_dict=bbox_dict,
        mode="test",
        transform=CVOGL_TRANSFORM,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    print("Creating model...")
    model = SampleGeoLite(emb_size=EMB_SIZE).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    best_loss = float("inf")
    print(f"Starting training for {args.max_epoch} epochs...")
    for epoch in range(args.max_epoch):
        train_loss = train_epoch(train_loader, model, optimizer, epoch, args.print_freq)

        val_loss = validate(val_loader, model, epoch)
        # top1, top5, top10 = evaluate(model, val_loader)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        # writer.add_scalar("Metrics/top1_acc", top1, epoch)
        # writer.add_scalar("Metrics/top5_acc", top5, epoch)
        # writer.add_scalar("Metrics/top10_acc", top10, epoch)

        print(
            f"Epoch {epoch + 1}/{args.max_epoch}:\t"
            f"Train Loss: {train_loss:.4f}\t"
            f"Val Loss: {val_loss:.4f}\t"
            # f"Top1: {top1:.4f}\t"
            # f"Top5: {top5:.4f}\t"
            # f"Top10: {top10:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(args.checkpoint, exist_ok=True)
            torch.save(model.state_dict(), f"{args.checkpoint}/best.pth")
            print(f"Saved best model with val_loss: {val_loss:.4f}")

    print(f"\nTraining complete. Best Val Loss: {best_loss:.4f}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SampleGeoLite Retrieval Training")
    parser.add_argument("--gpu", default="2", help="GPU id")
    parser.add_argument(
        "--num-workers", type=int, default=8, help="num workers for data loading"
    )
    parser.add_argument(
        "--max-epoch", type=int, default=NUM_EPOCHS, help="training epoch"
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="image size")
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

    os.makedirs(args.checkpoint, exist_ok=True)

    # main(args)
    eval_model(run=True, checkpoint_path="/data/feihong/ckpt/retrieval_sample4geo/best.pth")
    eval_model(run=False)
