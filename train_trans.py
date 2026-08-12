#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official-style dual-DeiT TransGeo retrieval with an attached grounding head."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bbox.yolo_utils import bbox_iou
from dataset import DEFAULT_SAT_TARGET_SIZE, ShiftedSatelliteDroneDataset
from grounding.utils.utils import AverageMeter
from train_uni import (
    BBOX_IOU_LOSS_WEIGHT,
    HEATMAP_SIGMA,
    TRAIN_BBOX_SCALE,
    TRAIN_CROP_RATIO_RANGE,
    DummyTokenizer,
    IdentityBalancedBatchSampler,
    RetrievalMemoryQueue,
    TransformProcessorWrapper,
    bbox_regression_loss,
    decode_bbox,
    multi_positive_cross_entropy,
    resume_training_checkpoint,
    retrieval_loss_with_memory,
    save_training_checkpoint,
    spatial_ce_loss,
    validate as validate_joint,
)
from unify_compare_dataset import (
    JointGeoTrainingDataset,
    UnifiedSiglipSuppComparisonDataset,
)


# --- Configuration ---
# Keep the comparison input geometry aligned with unified_siglip_supp.py:
# SigLIP2's default query processor produces 224x224 tensors and Config.UNIV_SAT_SIZE
# produces 768x432 satellite tensors.  Normalization remains ImageNet-style because
# that is the preprocessing expected by the ImageNet-pretrained DeiT backbone.
DRONE_SIZE = (224, 224)  # (width, height)
SAT_SIZE = (DEFAULT_SAT_TARGET_SIZE[1], DEFAULT_SAT_TARGET_SIZE[0])  # (width, height)
BATCH_SIZE = 16
GRAD_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 20
LEARNING_RATE = 5e-5
BACKBONE_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 0
MIN_LEARNING_RATE = 1e-10
PRINT_FREQ = 50
PROJECTION_DIM = 1000
DETAIL_DIM = 384
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6
PATCH_SIZE = 16
TRANS_BACKBONE_NAME = "deit_small_distilled_patch16_224"
RETRIEVAL_LOSS_WEIGHT = 1.0
TRIPLET_LOSS_WEIGHT = 1.0
LOCALIZATION_LOSS_WEIGHT = 1.0
BBOX_LOSS_WEIGHT = 5.0
RERANK_LOSS_WEIGHT = 1.0
MEMORY_QUEUE_SIZE = 512
MEMORY_QUEUE_START_EPOCH = 10
RETRIEVAL_ONLY_EPOCHS = 8
FREEZE_BACKBONE_EPOCHS = 0
AMP_DTYPE = "bf16"
VAL_FRACTION = 1.0
DATA_PROTOCOL = "unified_siglip_supp"
EVAL_SPLIT = "val"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class TransGeoVisionBranch(nn.Module):
    """One of the two independent DeiT-S/16 encoders used by TransGeo."""

    def __init__(
        self,
        image_size: Tuple[int, int],
        output_dim: int,
        backbone_name: str = TRANS_BACKBONE_NAME,
        pretrained: bool = False,
    ):
        super().__init__()
        image_h, image_w = int(image_size[0]), int(image_size[1])
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=bool(pretrained),
            # Prefer the official DeiT URL over Hugging Face so the first run
            # follows the source implementation and uses torch's local cache.
            pretrained_cfg_overlay={"hf_hub_id": None},
            img_size=(image_h, image_w),
            num_classes=int(output_dim),
        )
        self.embed_dim = int(self.backbone.embed_dim)
        patch_size = self.backbone.patch_embed.patch_size
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.num_prefix_tokens = int(self.backbone.num_prefix_tokens)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone.forward_features(images)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise RuntimeError(
                f"Expected DeiT token tensor shaped (B,N,C), got {type(tokens)} "
                f"with shape={getattr(tokens, 'shape', None)}."
            )
        # Official TransGeo uses the averaged DeiT class/distillation heads as
        # its metric-learning embedding (1000 dimensions in the paper).
        global_feature = self.backbone.forward_head(tokens, pre_logits=False)
        patch_tokens = tokens[:, self.num_prefix_tokens :]
        grid_h = images.shape[-2] // self.patch_size[0]
        grid_w = images.shape[-1] // self.patch_size[1]
        expected_tokens = int(grid_h * grid_w)
        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"DeiT patch count mismatch: got {patch_tokens.shape[1]}, "
                f"expected {grid_h}x{grid_w}={expected_tokens}."
            )
        patch_map = patch_tokens.transpose(1, 2).reshape(
            images.shape[0], self.embed_dim, grid_h, grid_w
        )
        return global_feature, patch_map


class TransGeoGrounding(nn.Module):
    def __init__(
        self,
        sat_size: Tuple[int, int] = SAT_SIZE,
        drone_size: Tuple[int, int] = DRONE_SIZE,
        proj_dim: int = PROJECTION_DIM,
        detail_dim: int = DETAIL_DIM,
        embed_dim: int = EMBED_DIM,
        depth: int = DEPTH,
        num_heads: int = NUM_HEADS,
        patch_size: int = PATCH_SIZE,
        backbone_name: str = TRANS_BACKBONE_NAME,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        if (int(embed_dim), int(depth), int(num_heads), int(patch_size)) != (
            EMBED_DIM,
            DEPTH,
            NUM_HEADS,
            PATCH_SIZE,
        ):
            raise ValueError(
                "Official TransGeo DeiT-S/16 requires embed_dim=384, depth=12, "
                "num_heads=6, patch_size=16."
            )
        sat_hw = (int(sat_size[1]), int(sat_size[0]))
        drone_hw = (int(drone_size[1]), int(drone_size[0]))
        self.query_encoder = TransGeoVisionBranch(
            image_size=drone_hw,
            output_dim=proj_dim,
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
        )
        self.reference_encoder = TransGeoVisionBranch(
            image_size=sat_hw,
            output_dim=proj_dim,
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
        )
        self.query_global = nn.Identity()
        self.reference_global = nn.Identity()
        self.query_detail = nn.Sequential(
            nn.Conv2d(embed_dim, detail_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
        )
        self.reference_detail = nn.Sequential(
            nn.Conv2d(embed_dim, detail_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
        )
        self.bbox_head = nn.Sequential(
            nn.Conv2d(detail_dim * 2 + 1, detail_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, 4, kernel_size=1),
        )
        nn.init.zeros_(self.bbox_head[-1].weight)
        with torch.no_grad():
            self.bbox_head[-1].bias.copy_(
                torch.tensor([0.0, 0.0, -1.7346, -1.7346])
            )
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
        self.temperature = nn.Parameter(torch.tensor(0.07, dtype=torch.float32))
        self.heat_kernel_size = 3

    def _dynamic_heatmap(self, query_detail_map: torch.Tensor, aerial_detail: torch.Tensor) -> torch.Tensor:
        if query_detail_map.ndim != 4:
            raise ValueError(f"Expected query_detail_map shape (B, C, H, W), got {tuple(query_detail_map.shape)}.")
        if aerial_detail.ndim != 4:
            raise ValueError(f"Expected aerial_detail shape (B, C, H, W), got {tuple(aerial_detail.shape)}.")
        if query_detail_map.shape[0] != aerial_detail.shape[0] or query_detail_map.shape[1] != aerial_detail.shape[1]:
            raise ValueError(
                f"Query/aerial detail mismatch: query={tuple(query_detail_map.shape)}, "
                f"aerial={tuple(aerial_detail.shape)}."
            )

        batch_size, channels, height, width = aerial_detail.shape
        kernel = F.adaptive_avg_pool2d(
            query_detail_map,
            (self.heat_kernel_size, self.heat_kernel_size),
        )
        kernel = kernel - kernel.mean(dim=(2, 3), keepdim=True)
        kernel = F.normalize(kernel.flatten(1), p=2, dim=1).view_as(kernel)
        sat_features = F.normalize(aerial_detail.contiguous(), p=2, dim=1)
        conv_input = sat_features.reshape(1, batch_size * channels, height, width)
        conv_kernel = kernel.reshape(batch_size, channels, self.heat_kernel_size, self.heat_kernel_size)
        heatmap = F.conv2d(
            conv_input,
            conv_kernel,
            padding=self.heat_kernel_size // 2,
            groups=batch_size,
        )
        return heatmap.view(batch_size, 1, height, width)

    def forward(
        self,
        query_imgs,
        aerial_imgs,
        compute_grounding: bool = True,
        compute_rerank: bool = True,
    ):
        query_embedding, query_patch_map = self.query_encoder(query_imgs)
        aerial_embedding, aerial_patch_map = self.reference_encoder(aerial_imgs)

        with torch.autocast(device_type=query_embedding.device.type, enabled=False):
            query_global = F.normalize(
                self.query_global(query_embedding.float()),
                p=2,
                dim=1,
            )
            aerial_global = F.normalize(
                self.reference_global(aerial_embedding.float()),
                p=2,
                dim=1,
            )
            scale = self.logit_scale.float().exp().clamp(max=100.0)
            retrieval_logits = scale * query_global @ aerial_global.t()

        outputs = {
            "query_global": query_global,
            "aerial_global": aerial_global,
            "logit_scale": scale,
            "retrieval_logits": retrieval_logits,
        }
        if not compute_grounding:
            return outputs

        query_detail_map = F.normalize(self.query_detail(query_patch_map).float(), p=2, dim=1)
        aerial_detail = F.normalize(self.reference_detail(aerial_patch_map).float(), p=2, dim=1)
        ground_detail = F.normalize(
            F.adaptive_avg_pool2d(query_detail_map, 1).flatten(1),
            p=2,
            dim=1,
        )

        temp = self.temperature.float().clamp(min=0.03, max=0.2)
        with torch.autocast(device_type=query_embedding.device.type, enabled=False):
            heatmap_logits = self._dynamic_heatmap(query_detail_map, aerial_detail) / temp
        heat_gate = F.softmax(heatmap_logits.flatten(1), dim=1).view_as(heatmap_logits)
        heat_gate = heat_gate * float(heat_gate.shape[-2] * heat_gate.shape[-1])
        query_map = ground_detail[:, :, None, None].expand_as(aerial_detail)
        bbox_raw = self.bbox_head(
            torch.cat([aerial_detail, query_map, heat_gate], dim=1)
        ).float()

        outputs.update({
            "ground_detail": ground_detail,
            "aerial_detail": aerial_detail,
            "heatmap_logits": heatmap_logits,
            "bbox_raw": bbox_raw,
        })
        if compute_rerank:
            with torch.autocast(device_type=query_embedding.device.type, enabled=False):
                detail_scores = (
                    torch.einsum("bd,kdhw->bkhw", ground_detail, aerial_detail)
                    / temp
                )
                detail_logits = detail_scores.flatten(2).max(dim=-1).values
                # Inference follows the existing global+detail reranking rule. The
                # training loss supervises detail_logits alone so the global
                # branch cannot mask a weak local matching branch.
                outputs["detail_logits"] = detail_logits
                outputs["rerank_logits"] = retrieval_logits + detail_logits
        return outputs

    def bbox_forward(self, query_imgs, aerial_imgs):
        outputs = self.forward(query_imgs, aerial_imgs)
        return decode_bbox(
            outputs["heatmap_logits"],
            outputs["bbox_raw"],
            image_wh=(aerial_imgs.shape[-1], aerial_imgs.shape[-2]),
        )


def exhaustive_soft_margin_triplet(
    query_feats: torch.Tensor,
    aerial_feats: torch.Tensor,
    identities: torch.Tensor,
    alpha: float = 10.0,
) -> torch.Tensor:
    """Official TransGeo-style exhaustive bidirectional soft-margin loss."""
    with torch.autocast(device_type=query_feats.device.type, enabled=False):
        query_feats = F.normalize(query_feats.float(), p=2, dim=1)
        aerial_feats = F.normalize(aerial_feats.float(), p=2, dim=1)
        similarity = query_feats @ aerial_feats.t()
    identities = identities.view(-1).long()
    positive_mask = identities[:, None].eq(identities[None, :])

    def directional_loss(scores: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        row_losses: List[torch.Tensor] = []
        for row_idx in range(scores.shape[0]):
            positive_scores = scores[row_idx][positives[row_idx]]
            negative_scores = scores[row_idx][~positives[row_idx]]
            if positive_scores.numel() == 0 or negative_scores.numel() == 0:
                continue
            differences = negative_scores[:, None] - positive_scores[None, :]
            row_losses.append(F.softplus(float(alpha) * differences).mean())
        if not row_losses:
            return scores.new_zeros(())
        return torch.stack(row_losses).mean()

    return 0.5 * (
        directional_loss(similarity, positive_mask)
        + directional_loss(similarity.t(), positive_mask.t())
    )


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    gt_bbox,
    image_wh,
    args,
    epoch: int = 0,
    identities: torch.Tensor = None,
    memory_queue: RetrievalMemoryQueue = None,
):
    if identities is None:
        identities = torch.arange(outputs["retrieval_logits"].shape[0], device=gt_bbox.device)
    identities = identities.view(-1).long()
    retrieval_loss = retrieval_loss_with_memory(
        outputs["query_global"],
        outputs["aerial_global"],
        outputs["logit_scale"],
        memory_queue,
        label_smoothing=args.label_smoothing,
        identities=identities,
    )
    triplet_loss = exhaustive_soft_margin_triplet(
        outputs["query_global"],
        outputs["aerial_global"],
        identities,
        alpha=args.triplet_alpha,
    )
    if epoch < args.retrieval_only_epochs:
        grounding_scale = 0.0
        localization_loss = retrieval_loss.new_zeros(())
        bbox_loss = retrieval_loss.new_zeros(())
        bbox_l1 = retrieval_loss.new_zeros(())
        bbox_iou = retrieval_loss.new_zeros(())
        rerank_loss = retrieval_loss.new_zeros(())
    else:
        grounding_scale = min(
            1.0,
            (epoch - args.retrieval_only_epochs + 1) / max(float(args.grounding_ramp_epochs), 1.0),
        )
        localization_loss = spatial_ce_loss(
            outputs["heatmap_logits"],
            gt_bbox,
            image_wh,
            sigma=args.heatmap_sigma,
        )
        bbox_loss, bbox_l1, bbox_iou = bbox_regression_loss(
            outputs["heatmap_logits"],
            outputs["bbox_raw"],
            gt_bbox,
            image_wh,
            iou_weight=args.bbox_iou_loss_weight,
        )
        rerank_loss = 0.5 * (
            multi_positive_cross_entropy(outputs["detail_logits"], identities, identities)
            + multi_positive_cross_entropy(outputs["detail_logits"].t(), identities, identities)
        )
    total = (
        args.retrieval_loss_weight * retrieval_loss
        + args.triplet_loss_weight * triplet_loss
        + grounding_scale * args.localization_loss_weight * localization_loss
        + grounding_scale * args.bbox_loss_weight * bbox_loss
        + grounding_scale * args.rerank_loss_weight * rerank_loss
    )
    return total, {
        "retrieval": retrieval_loss,
        "triplet": triplet_loss,
        "localization": localization_loss,
        "bbox": bbox_loss,
        "bbox_l1": bbox_l1,
        "bbox_iou": bbox_iou,
        "rerank": rerank_loss,
        "grounding_scale": bbox_loss.new_tensor(grounding_scale),
    }


@torch.no_grad()
def asam_perturb(model: nn.Module, rho: float) -> List[Tuple[nn.Parameter, torch.Tensor]]:
    """Apply the adaptive SAM ascent step used by the official TransGeo code."""
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not parameters or float(rho) <= 0.0:
        return []
    shared_device = parameters[0].device
    grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                (parameter.detach().abs() * parameter.grad.detach()).norm(p=2).to(shared_device)
                for parameter in parameters
            ]
        ),
        ord=2,
    )
    scale = float(rho) / (grad_norm + 1e-12)
    perturbations: List[Tuple[nn.Parameter, torch.Tensor]] = []
    for parameter in parameters:
        perturbation = parameter.detach().square() * parameter.grad.detach() * scale.to(parameter)
        parameter.add_(perturbation)
        perturbations.append((parameter, perturbation))
    return perturbations


@torch.no_grad()
def asam_restore(perturbations: List[Tuple[nn.Parameter, torch.Tensor]]) -> None:
    for parameter, perturbation in perturbations:
        parameter.sub_(perturbation)


def losses_are_finite(loss: torch.Tensor, loss_items: Dict[str, torch.Tensor]) -> bool:
    if not bool(torch.isfinite(loss.detach())):
        return False
    return all(bool(torch.isfinite(value.detach())) for value in loss_items.values())


@torch.no_grad()
def clamp_trainable_temperatures(model: nn.Module) -> None:
    model.logit_scale.clamp_(min=0.0, max=float(np.log(100.0)))
    model.temperature.clamp_(min=0.03, max=0.2)


@torch.no_grad()
def average_accumulated_gradients(model: nn.Module, microbatch_count: int) -> None:
    """Average gradients over the completed accumulation window."""
    divisor = float(max(int(microbatch_count), 1))
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(divisor)


def train_epoch(loader, model, optimizer, epoch, writer, args, memory_queue=None, scaler=None):
    model.train()
    if epoch < args.freeze_backbone_epochs or args.freeze_geoformer:
        model.query_encoder.eval()
        model.reference_encoder.eval()
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)
    grounding_active = epoch >= args.retrieval_only_epochs
    queue_active = epoch >= args.memory_queue_start_epoch
    active_memory_queue = memory_queue if queue_active else None
    meters = {
        "loss": AverageMeter(),
        "retrieval": AverageMeter(),
        "triplet": AverageMeter(),
        "localization": AverageMeter(),
        "bbox": AverageMeter(),
        "bbox_l1": AverageMeter(),
        "bbox_iou": AverageMeter(),
        "rerank": AverageMeter(),
        "grounding_scale": AverageMeter(),
        "nonfinite_batches": AverageMeter(),
    }
    batch_time = AverageMeter()
    end = time.time()
    consecutive_nonfinite = 0
    accumulation_steps = max(1, int(args.grad_accumulation_steps))
    accumulated_microbatches = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        global_step = epoch * len(loader) + batch_idx
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        aerial_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        gt_bbox = batch["bbox"].to(DEVICE, non_blocking=True)
        identities = batch["satellite_id"].to(DEVICE, non_blocking=True)
        image_wh = (aerial_imgs.shape[-1], aerial_imgs.shape[-2])

        amp_enabled = bool(args.amp and torch.cuda.is_available())
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                query_imgs,
                aerial_imgs,
                compute_grounding=grounding_active,
            )
            loss, loss_items = compute_losses(
                outputs,
                gt_bbox,
                image_wh,
                args,
                epoch=epoch,
                identities=identities,
                memory_queue=active_memory_queue,
            )
        if not losses_are_finite(loss, loss_items):
            consecutive_nonfinite += 1
            meters["nonfinite_batches"].update(1.0, 1)
            optimizer.zero_grad(set_to_none=True)
            accumulated_microbatches = 0
            bad_terms = [
                name for name, value in loss_items.items() if not bool(torch.isfinite(value.detach()))
            ]
            print(
                f"Warning: skipped non-finite batch epoch={epoch} step={batch_idx}; "
                f"terms={bad_terms or ['total']}"
            )
            if writer is not None:
                writer.add_scalar("TrainStep/nonfinite_batch", 1.0, global_step)
            if consecutive_nonfinite >= args.max_consecutive_nonfinite:
                raise FloatingPointError(
                    f"Stopped after {consecutive_nonfinite} consecutive non-finite batches."
                )
            continue
        consecutive_nonfinite = 0
        meters["nonfinite_batches"].update(0.0, 1)
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        accumulated_microbatches += 1
        if args.asam_rho > 0.0:
            gradients_finite = True
            if scaler is not None and scaler.is_enabled():
                inverse_scale = 1.0 / float(scaler.get_scale())
                finite_flag = torch.ones((), dtype=torch.bool, device=query_imgs.device)
                for parameter in model.parameters():
                    if parameter.grad is None:
                        continue
                    parameter.grad.mul_(inverse_scale)
                    finite_flag.logical_and_(torch.isfinite(parameter.grad).all())
                gradients_finite = bool(finite_flag.item())
            perturbations = asam_perturb(model, args.asam_rho) if gradients_finite else []
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    outputs_perturbed = model(
                        query_imgs,
                        aerial_imgs,
                        compute_grounding=grounding_active,
                    )
                    loss_perturbed, _ = compute_losses(
                        outputs_perturbed,
                        gt_bbox,
                        image_wh,
                        args,
                        epoch=epoch,
                        identities=identities,
                        memory_queue=active_memory_queue,
                    )
                if scaler is None:
                    loss_perturbed.backward()
                else:
                    scaler.scale(loss_perturbed).backward()
            finally:
                asam_restore(perturbations)
        should_update = (
            accumulated_microbatches >= accumulation_steps
            or batch_idx + 1 == len(loader)
        )
        if should_update:
            scaler_enabled = scaler is not None and scaler.is_enabled()
            if scaler_enabled:
                scaler.unscale_(optimizer)

            # Average the accumulated gradients explicitly. This also handles
            # the shorter final accumulation window without shrinking its step.
            average_accumulated_gradients(model, accumulated_microbatches)

            max_grad_norm = args.grad_clip if args.grad_clip > 0 else float("inf")
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            gradients_finite = bool(torch.isfinite(grad_norm))
            if gradients_finite:
                if scaler_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                clamp_trainable_temperatures(model)
            else:
                if scaler_enabled:
                    # Do not call scaler.step here: a norm can overflow even
                    # when individual gradients were finite at unscale time.
                    reduced_scale = max(float(scaler.get_scale()) * 0.5, 1.0)
                    scaler.update(new_scale=reduced_scale)
                print(
                    "Warning: skipped optimizer update with non-finite grad "
                    f"norm at step={global_step}."
                )
                if writer is not None:
                    writer.add_scalar("TrainStep/nonfinite_gradient", 1.0, global_step)
            optimizer.zero_grad(set_to_none=True)
            accumulated_microbatches = 0
        if active_memory_queue is not None:
            active_memory_queue.enqueue(
                outputs["query_global"],
                outputs["aerial_global"],
                identities,
            )

        bs = query_imgs.shape[0]
        meters["loss"].update(loss.item(), bs)
        for name, value in loss_items.items():
            meters[name].update(value.item(), bs)
        batch_time.update(time.time() - end)
        end = time.time()

        if writer is not None:
            writer.add_scalar("TrainStep/loss", loss.item(), global_step)
            for name, value in loss_items.items():
                writer.add_scalar(f"TrainStep/{name}", value.item(), global_step)
            writer.add_scalar("lr/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        if batch_idx % args.print_freq == 0:
            print(
                f"Epoch: [{epoch}][{batch_idx}/{len(loader)}]\t"
                f"Time: {batch_time.val:.3f}\t"
                f"Loss: {meters['loss'].val:.4f} ({meters['loss'].avg:.4f})\t"
                f"Ret: {meters['retrieval'].val:.4f}\t"
                f"Tri: {meters['triplet'].val:.4f}\t"
                f"Loc: {meters['localization'].val:.4f}\t"
                f"Bbox: {meters['bbox'].val:.4f}\t"
                f"Rerank: {meters['rerank'].val:.4f}"
            )

    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def validate_global_retrieval(loader, model) -> float:
    """Compute global-only R@1 while leaving the shared joint validator unchanged."""
    model.eval()
    query_features: List[torch.Tensor] = []
    query_ids: List[torch.Tensor] = []
    gallery_features: Dict[int, torch.Tensor] = {}
    for batch in tqdm(loader, desc="Validating global retrieval"):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        aerial_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        identities = batch["satellite_id"].view(-1).long()
        outputs = model(query_imgs, aerial_imgs, compute_grounding=False)
        query_features.append(outputs["query_global"].detach().cpu())
        query_ids.append(identities.cpu())
        aerial_global = outputs["aerial_global"].detach().cpu()
        for idx, identity in enumerate(identities.tolist()):
            if identity not in gallery_features:
                gallery_features[identity] = aerial_global[idx]

    if not query_features or not gallery_features:
        return 0.0
    gallery_ids = sorted(gallery_features)
    gallery_tensor = torch.stack([gallery_features[identity] for identity in gallery_ids]).to(DEVICE)
    gallery_id_tensor = torch.tensor(gallery_ids, dtype=torch.long, device=DEVICE)
    query_tensor = torch.cat(query_features)
    query_id_tensor = torch.cat(query_ids)
    hits: List[torch.Tensor] = []
    for start in range(0, query_tensor.shape[0], 64):
        scores = query_tensor[start : start + 64].to(DEVICE) @ gallery_tensor.t()
        predicted = gallery_id_tensor[scores.argmax(dim=1)].cpu()
        hits.append(predicted.eq(query_id_tensor[start : start + 64]))
    return float(torch.cat(hits).float().mean().item())


def validate(loader, model):
    metrics = validate_joint(loader, model)
    metrics["retrieval_rerank_top1"] = float(metrics["retrieval_top1"])
    metrics["retrieval_global_top1"] = validate_global_retrieval(loader, model)
    return metrics


def build_dataloaders(args):
    processor = TransformProcessorWrapper(DRONE_SIZE)
    processor_sat = TransformProcessorWrapper(args.sat_size)
    tokenizer = DummyTokenizer()

    common_dataset_kwargs = {
        "processor": processor,
        "processor_sat": processor_sat,
        "tokenizer": tokenizer,
        "subset_heights": args.heights,
        "subset_angles": args.angles,
    }
    if args.data_protocol == "unified_siglip_supp":
        # Exact sample/crop/bbox contract used by unified_siglip_supp.py.
        train_dataset = UnifiedSiglipSuppComparisonDataset(
            split="train",
            **common_dataset_kwargs,
        )
        eval_dataset = UnifiedSiglipSuppComparisonDataset(
            split=args.eval_split,
            test_crop_ratio=1.0,
            **common_dataset_kwargs,
        )
    elif args.data_protocol == "joint_geo_legacy":
        train_dataset = JointGeoTrainingDataset(
            train_crop_ratio_range=(args.train_crop_min_ratio, args.train_crop_max_ratio),
            train_bbox_scale=args.train_bbox_scale,
            **common_dataset_kwargs,
        )
        eval_dataset = ShiftedSatelliteDroneDataset(
            split=args.eval_split,
            test_crop_ratio=1.0,
            **common_dataset_kwargs,
        )
    else:
        raise ValueError(f"Unsupported data protocol: {args.data_protocol}")

    raw_eval_count = len(eval_dataset)
    val_fraction = float(args.val_fraction)
    if val_fraction <= 0.0 or val_fraction > 1.0:
        raise ValueError(f"Expected --val-fraction in (0, 1], got {val_fraction}.")
    if val_fraction < 1.0:
        step = max(int(round(1.0 / val_fraction)), 1)
        eval_indices = list(range(0, raw_eval_count, step))
        eval_dataset = Subset(eval_dataset, eval_indices)

    loader_kwargs = {
        "pin_memory": torch.cuda.is_available(),
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
        "prefetch_factor": 4 if args.num_workers > 0 else None,
    }
    if args.identity_balanced_batches:
        batch_sampler = IdentityBalancedBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            drop_last=True,
            seed=args.seed,
        )
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        # This is the loader behavior used by unified_siglip_supp.py.
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    return train_loader, eval_loader, len(train_dataset), len(eval_dataset)


def data_protocol_summary(args) -> Dict[str, object]:
    """Return the auditable data/training contract stored with each run."""
    unified = args.data_protocol == "unified_siglip_supp"
    return {
        "name": args.data_protocol,
        "reference": "unified_siglip_supp.py" if unified else "legacy joint Geo adapter",
        "train_split": "train",
        "checkpoint_selection_split": args.eval_split,
        "final_test_split": "test (through test_unify.py)",
        "train_crop_ratio_range": None if unified else [
            float(args.train_crop_min_ratio),
            float(args.train_crop_max_ratio),
        ],
        "train_bbox_scale": 1.0 if unified else float(args.train_bbox_scale),
        "sampler": "identity_balanced" if args.identity_balanced_batches else "shuffle",
        "heights": list(args.heights),
        "angles": list(args.angles),
        "drone_size_wh": list(DRONE_SIZE),
        "satellite_size_wh": list(args.sat_size),
        "normalization": "ImageNet (DeiT pretrained contract)",
        "rerank_training_score": "detail_only",
        "rerank_inference_score": "global_plus_detail",
        "micro_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.grad_accumulation_steps),
        "effective_batch_size": int(args.batch_size * args.grad_accumulation_steps),
    }

def adjust_learning_rate(args, optimizer, epoch):
    for param_group in optimizer.param_groups:
        base_lr = float(param_group.get("base_lr", args.lr))
        min_lr = float(args.min_lr) * base_lr / max(float(args.lr), 1e-12)
        if epoch < args.warmup_epochs:
            lr = base_lr * float(epoch + 1) / max(float(args.warmup_epochs), 1.0)
        else:
            progress = (epoch - args.warmup_epochs) / max(float(args.max_epoch - args.warmup_epochs), 1.0)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))
        param_group["lr"] = lr
    print(("lr", [param_group["lr"] for param_group in optimizer.param_groups]))


def set_transgeo_backbones_trainable(model: nn.Module, trainable: bool) -> None:
    for encoder in (model.query_encoder, model.reference_encoder):
        for param in encoder.parameters():
            param.requires_grad = bool(trainable)


def main(args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.amp and args.amp_dtype == "bf16" and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            print("CUDA device does not support BF16; disabling AMP for TransGeo stability.")
            args.amp = False

    os.makedirs(args.checkpoint, exist_ok=True)
    writer = SummaryWriter(f"runs/{args.savename}")
    protocol = data_protocol_summary(args)
    args.data_contract = protocol

    print(
        "Creating datasets from shared data source with protocol="
        f"{args.data_protocol}..."
    )
    train_loader, val_loader, train_count, val_count = build_dataloaders(args)
    protocol["train_num_samples"] = int(train_count)
    protocol["checkpoint_selection_num_samples"] = int(val_count)
    train_config = dict(vars(args))
    with open(os.path.join(args.checkpoint, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)
    print(
        f"Found {train_count} training samples, {val_count} {args.eval_split} samples; "
        f"sampler={protocol['sampler']}, effective_batch_size="
        f"{protocol['effective_batch_size']} "
        f"({args.batch_size} x {args.grad_accumulation_steps})."
    )

    print("Creating official-style dual-DeiT TransGeoGrounding model...")
    model = TransGeoGrounding(
        sat_size=args.sat_size,
        drone_size=DRONE_SIZE,
        proj_dim=args.proj_dim,
        detail_dim=args.detail_dim,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        backbone_name=args.backbone_name,
        pretrained_backbone=args.pretrained_backbone,
    ).to(DEVICE)
    print(
        f"Backbone={args.backbone_name}, ImageNet pretrained={args.pretrained_backbone}, "
        f"retrieval-only epochs={args.retrieval_only_epochs}, "
        f"memory queue starts at epoch={args.memory_queue_start_epoch} "
        f"with capacity={args.memory_queue_size}."
    )
    set_transgeo_backbones_trainable(
        model,
        not (args.freeze_geoformer or args.freeze_backbone_epochs > 0),
    )
    if args.freeze_geoformer:
        print("Both DeiT backbones stay frozen for the entire run.")
    elif args.freeze_backbone_epochs > 0:
        print(f"Both DeiT backbones are frozen for the first {args.freeze_backbone_epochs} epochs.")

    backbone_params = list(model.query_encoder.parameters()) + list(model.reference_encoder.parameters())
    backbone_param_ids = {id(param) for param in backbone_params}
    head_params = [param for param in model.parameters() if id(param) not in backbone_param_ids]
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr, "base_lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr, "base_lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    print(
        f"Optimizer param groups: dual DeiT lr={args.backbone_lr:.2e}, "
        f"heads lr={args.lr:.2e}"
    )
    memory_queue = RetrievalMemoryQueue(args.memory_queue_size) if args.memory_queue_size > 0 else None
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(args.amp and args.amp_dtype == "fp16" and torch.cuda.is_available()),
    )
    start_epoch = 0
    best_retrieval = -1.0
    best_iou = -1.0
    best_uiou = -1.0
    if args.resume:
        start_epoch, best_iou, best_uiou = resume_training_checkpoint(args.resume, model, optimizer)
        print(f"Resumed {args.resume} at epoch {start_epoch}.")

    print(f"Starting training for {args.max_epoch} epochs...")
    for epoch in range(start_epoch, args.max_epoch):
        backbone_trainable = not args.freeze_geoformer and epoch >= args.freeze_backbone_epochs
        set_transgeo_backbones_trainable(model, backbone_trainable)
        adjust_learning_rate(args, optimizer, epoch)
        train_metrics = train_epoch(
            train_loader,
            model,
            optimizer,
            epoch,
            writer,
            args,
            memory_queue=memory_queue,
            scaler=scaler,
        )
        val_metrics = validate(val_loader, model)

        for name, value in train_metrics.items():
            writer.add_scalar(f"Train/{name}", value, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f"Val/{name}", value, epoch)

        print(
            f"Epoch {epoch + 1}/{args.max_epoch}:\t"
            f"Train Loss: {train_metrics['loss']:.4f}\t"
            f"Ret: {train_metrics['retrieval']:.4f}\t"
            f"Tri: {train_metrics['triplet']:.4f}\t"
            f"Loc: {train_metrics['localization']:.4f}\t"
            f"Bbox: {train_metrics['bbox']:.4f}\t"
            f"Rerank: {train_metrics['rerank']:.4f}\t"
            f"Val mIoU: {val_metrics['mean_iou']:.4f}\t"
            f"Val global R@1: {val_metrics['retrieval_global_top1']:.4f}\t"
            f"Val rerank R@1: {val_metrics['retrieval_rerank_top1']:.4f}\t"
            f"Val uIoU: {val_metrics['uiou']:.4f}"
        )

        grounding_active = epoch >= args.retrieval_only_epochs
        improved_retrieval = val_metrics["retrieval_global_top1"] > best_retrieval
        improved_iou = grounding_active and val_metrics["mean_iou"] > best_iou
        improved_uiou = grounding_active and val_metrics["uiou"] > best_uiou
        if improved_retrieval:
            best_retrieval = val_metrics["retrieval_global_top1"]
        if improved_iou:
            best_iou = val_metrics["mean_iou"]
        if improved_uiou:
            best_uiou = val_metrics["uiou"]
        save_training_checkpoint(
            os.path.join(args.checkpoint, "last.pth"),
            model,
            optimizer,
            epoch,
            best_iou,
            best_uiou,
            args,
        )
        if improved_retrieval:
            save_training_checkpoint(
                os.path.join(args.checkpoint, "best_retrieval.pth"),
                model,
                optimizer,
                epoch,
                best_iou,
                best_uiou,
                args,
            )
        if improved_iou:
            save_training_checkpoint(
                os.path.join(args.checkpoint, "best_iou.pth"),
                model,
                optimizer,
                epoch,
                best_iou,
                best_uiou,
                args,
            )
        if improved_uiou:
            save_training_checkpoint(
                os.path.join(args.checkpoint, "best_joint.pth"),
                model,
                optimizer,
                epoch,
                best_iou,
                best_uiou,
                args,
            )

    print("\nTraining complete. Saved checkpoint to last.pth")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual-DeiT TransGeo retrieval with staged grounding training"
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-epoch", type=int, default=NUM_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LEARNING_RATE)
    parser.add_argument("--min-lr", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=GRAD_ACCUMULATION_STEPS,
        help=(
            "micro-batches per optimizer step; default 16x4=64 matches the "
            "effective batch size of unified_siglip_supp baseline (32x2)"
        ),
    )
    parser.add_argument(
        "--data-protocol",
        choices=["unified_siglip_supp", "joint_geo_legacy"],
        default=DATA_PROTOCOL,
        help=(
            "dataset contract; the default exactly matches unified_siglip_supp "
            "sample, crop, bbox and shuffle semantics"
        ),
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default=EVAL_SPLIT,
        help=(
            "split used only for checkpoint selection; default=val keeps the "
            "shared test split untouched until test_unify.py"
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=VAL_FRACTION,
        help="fraction of the validation split evaluated each epoch",
    )
    parser.add_argument(
        "--sat-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=list(SAT_SIZE),
        help="satellite image size as WIDTH HEIGHT",
    )
    parser.add_argument("--savename", type=str, default="trans_geo_unified_siglip_fair")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--print-freq", type=int, default=PRINT_FREQ)
    parser.add_argument("--proj-dim", type=int, default=PROJECTION_DIM)
    parser.add_argument("--detail-dim", type=int, default=DETAIL_DIM)
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="InfoNCE smoothing; 0.1 follows the paper setting from the diagnostic report",
    )
    parser.add_argument("--heatmap-sigma", type=float, default=HEATMAP_SIGMA)
    parser.add_argument("--retrieval-loss-weight", type=float, default=RETRIEVAL_LOSS_WEIGHT)
    parser.add_argument("--localization-loss-weight", type=float, default=LOCALIZATION_LOSS_WEIGHT)
    parser.add_argument("--bbox-loss-weight", type=float, default=BBOX_LOSS_WEIGHT)
    parser.add_argument("--rerank-loss-weight", type=float, default=RERANK_LOSS_WEIGHT)
    parser.add_argument("--triplet-loss-weight", type=float, default=TRIPLET_LOSS_WEIGHT)
    parser.add_argument("--triplet-alpha", type=float, default=10.0)
    parser.add_argument("--bbox-iou-loss-weight", type=float, default=BBOX_IOU_LOSS_WEIGHT)
    parser.add_argument(
        "--retrieval-only-epochs",
        type=int,
        default=RETRIEVAL_ONLY_EPOCHS,
        help="train only global retrieval before enabling grounding and reranking",
    )
    parser.add_argument("--grounding-ramp-epochs", type=int, default=3)
    parser.add_argument("--memory-queue-size", type=int, default=MEMORY_QUEUE_SIZE)
    parser.add_argument(
        "--memory-queue-start-epoch",
        type=int,
        default=MEMORY_QUEUE_START_EPOCH,
        help="delay cross-batch memory until global retrieval has warmed up",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--max-consecutive-nonfinite",
        type=int,
        default=3,
        help="abort instead of silently corrupting a run after N consecutive bad batches",
    )
    parser.add_argument(
        "--asam-rho",
        type=float,
        default=0.0,
        help="adaptive SAM radius from TransGeo; <=0 disables the second forward/backward pass",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=FREEZE_BACKBONE_EPOCHS,
        help="freeze both DeiT branches for the first N epochs",
    )
    parser.add_argument(
        "--identity-balanced-batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disabled by default to match unified_siglip_supp.py shuffle=True",
    )
    parser.add_argument("--train-crop-min-ratio", type=float, default=TRAIN_CROP_RATIO_RANGE[0])
    parser.add_argument("--train-crop-max-ratio", type=float, default=TRAIN_CROP_RATIO_RANGE[1])
    parser.add_argument("--train-bbox-scale", type=float, default=TRAIN_BBOX_SCALE)
    parser.add_argument(
        "--heights",
        type=int,
        nargs="+",
        default=[150, 200, 250, 300],
        help="drone heights used by training and checkpoint-selection splits",
    )
    parser.add_argument(
        "--angles",
        type=int,
        nargs="+",
        default=[0, 45, 90, 135, 180, 225, 270, 315],
        help="drone angles used by training and checkpoint-selection splits",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--amp-dtype",
        choices=["bf16", "fp16"],
        default=AMP_DTYPE,
        help="BF16 is the stable default for dual-DeiT training",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--backbone-name",
        type=str,
        default=TRANS_BACKBONE_NAME,
        help="timm DeiT/ViT backbone; the default matches official TransGeo",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="initialize both DeiT branches from ImageNet weights",
    )
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_geoformer",
        action="store_true",
        help="freeze both DeiT branches and train only task heads",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/data1/feihong/ckpt/trans_geo_unified_siglip_fair",
        help="Path to save model checkpoints",
    )
    args = parser.parse_args()
    args.sat_size = (int(args.sat_size[0]), int(args.sat_size[1]))
    args.heights = sorted({int(height) for height in args.heights})
    args.angles = sorted({int(angle) for angle in args.angles})
    if not args.heights or any(height not in {150, 200, 250, 300} for height in args.heights):
        parser.error("--heights must contain one or more of: 150 200 250 300")
    if args.retrieval_only_epochs < 0:
        parser.error("--retrieval-only-epochs must be >= 0")
    if args.memory_queue_start_epoch < 0:
        parser.error("--memory-queue-start-epoch must be >= 0")
    if args.grad_accumulation_steps <= 0:
        parser.error("--grad-accumulation-steps must be >= 1")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must satisfy 0 <= value < 1")
    if args.asam_rho > 0.0 and args.grad_accumulation_steps != 1:
        parser.error("ASAM currently requires --grad-accumulation-steps 1")
    if args.asam_rho > 0.0 and args.amp_dtype == "fp16":
        parser.error("ASAM with fp16 GradScaler is unsupported; use --amp-dtype bf16")
    valid_angles = {0, 45, 90, 135, 180, 225, 270, 315}
    if not args.angles or any(angle not in valid_angles for angle in args.angles):
        parser.error("--angles must contain one or more multiples of 45 from 0 to 315")
    main(args)
