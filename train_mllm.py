# --- Configuration ---
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForMultimodalLM = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForVision2Seq = None

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None

from dataset import DEFAULT_SUBSET_ANGLES, DEFAULT_SUBSET_HEIGHTS, ShiftedSatelliteDroneDataset
from test_unify import (
    FeatureBundle,
    _load_include_map,
    _path_label,
    group_summaries,
    print_summary,
    save_metrics,
    score_retrieval_and_uiou,
    summarize_records,
)
from unified_siglip_supp import (
    build_dataloader_kwargs,
    build_retrieval_soft_targets,
    info_nce_loss,
)


# --- Configuration ---
DEFAULT_MLLM_MODEL_NAME = "OpenGVLab/InternVL3_5-1B-HF"
DEFAULT_CACHE_DIR = "/media/data1/feihong/hf_cache"
DEFAULT_SAVE_DIR = "/media/data1/feihong/ckpt/mllm_retrieval"
DEFAULT_OUTPUT_DIR = "/media/data1/feihong/univerisity_dev/eval_results/test_unify"
DEFAULT_INCLUDE_FILE = "/media/data1/feihong/ckpt/include2.json"
PROJECTION_DIM = 768
SAT_SIZE = (432, 768)
IMAGE_PROMPT = " "


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dtype_from_name(dtype_name: str) -> Any:
    name = str(dtype_name).lower()
    if name == "auto":
        return "auto"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_hf_model_path(model_name: str, cache_dir: str, local_files_only: bool) -> str:
    model_path = Path(model_name)
    if model_path.is_dir():
        return str(model_path)
    if not local_files_only:
        return model_name

    repo_dir = Path(cache_dir) / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"No local snapshot found for {model_name} under {cache_dir}.")
    candidates = sorted(
        [path for path in repo_dir.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No local snapshot found for {model_name} under {repo_dir}.")
    return str(candidates[0])


def collate_mllm_batch(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "target_pixel_values",
        "search_pixel_values",
        "input_ids",
        "attention_mask",
        "index",
        "satellite_id",
        "bbox",
        "height",
        "angle",
    ]
    batch: Dict[str, Any] = {}
    for key in keys:
        if key in items[0]:
            values = [item[key] for item in items]
            if torch.is_tensor(values[0]):
                batch[key] = torch.stack(values)
            elif key in {"index", "satellite_id", "height", "angle"}:
                batch[key] = torch.as_tensor(values, dtype=torch.long)
            elif key == "bbox":
                batch[key] = torch.as_tensor(values, dtype=torch.float32)
            else:
                batch[key] = values
    for key in ["drone_path", "satellite_path", "query_image", "search_image"]:
        batch[key] = [item[key] for item in items]
    return batch


def open_images(images_or_paths: Sequence[Any]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for item in images_or_paths:
        if isinstance(item, Image.Image):
            images.append(item.convert("RGB"))
        else:
            images.append(Image.open(item).convert("RGB"))
    return images


class SimpleImageProcessor:
    def __init__(self, size: Tuple[int, int]):
        self.size = {"height": int(size[0]), "width": int(size[1])}

    def __call__(self, images: Image.Image, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        del return_tensors
        image = images.convert("RGB").resize((self.size["width"], self.size["height"]))
        array = np.asarray(image, dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return {"pixel_values": pixel_values}


class SimpleTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 64,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        tokens = str(text).split()[:max_length]
        length = max(1, len(tokens))
        input_ids = torch.zeros(1, max_length, dtype=torch.long)
        attention_mask = torch.zeros(1, max_length, dtype=torch.long)
        input_ids[0, :length] = torch.arange(1, length + 1, dtype=torch.long)
        attention_mask[0, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class MLLMQueryEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        proj_dim: int,
        dtype: str,
        load_4bit: bool,
        use_lora: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: str,
        local_files_only: bool,
    ):
        super().__init__()
        load_path = resolve_hf_model_path(model_name, cache_dir, local_files_only)
        self.processor = AutoProcessor.from_pretrained(
            load_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        model_cls = AutoModelForImageTextToText or AutoModelForMultimodalLM or AutoModelForVision2Seq or AutoModel
        model_kwargs: Dict[str, Any] = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "dtype": _dtype_from_name(dtype),
        }
        if load_4bit and torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        if load_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError("bitsandbytes/transformers BitsAndBytesConfig is required for --load-4bit.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        try:
            self.backbone = model_cls.from_pretrained(load_path, **model_kwargs)
        except TypeError:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self.backbone = model_cls.from_pretrained(load_path, **model_kwargs)

        if use_lora:
            if LoraConfig is None or get_peft_model is None:
                raise ImportError("peft is required for --use-lora.")
            if load_4bit and prepare_model_for_kbit_training is not None:
                self.backbone = prepare_model_for_kbit_training(self.backbone)
            target_modules = [item.strip() for item in lora_target_modules.split(",") if item.strip()]
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.hidden_dim = int(getattr(self.backbone.config, "hidden_size", 0))
        if self.hidden_dim <= 0:
            text_config = getattr(self.backbone.config, "text_config", None)
            self.hidden_dim = int(getattr(text_config, "hidden_size", 0))
        if self.hidden_dim <= 0:
            raise ValueError("Cannot infer MLLM hidden size from config.")
        self.projector = nn.Linear(self.hidden_dim, proj_dim)
        self.uses_device_map = getattr(self.backbone, "hf_device_map", None) is not None

    @property
    def input_device(self) -> torch.device:
        hf_device_map = getattr(self.backbone, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for device in hf_device_map.values():
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                if isinstance(device, str) and device.startswith("cuda"):
                    return torch.device(device)
        device = getattr(self.backbone, "device", None)
        if device is not None:
            return torch.device(device)
        return next(self.backbone.parameters()).device

    def _processor_inputs(self, texts: Sequence[str], images_or_paths: Sequence[Any]) -> Dict[str, torch.Tensor]:
        images = open_images(images_or_paths)
        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text},
                        ],
                    }
                ]
                for text, image in zip(texts, images)
            ]
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={"padding": True, "max_patches": 1},
                )
                return {
                    key: value.to(self.input_device)
                    for key, value in inputs.items()
                    if torch.is_tensor(value)
                }
            except TypeError:
                pass

        try:
            inputs = self.processor(text=list(texts), images=images, return_tensors="pt", padding=True)
        except TypeError:
            inputs = self.processor(list(texts), images, return_tensors="pt", padding=True)
        return {
            key: value.to(self.input_device)
            for key, value in inputs.items()
            if torch.is_tensor(value)
        }

    def _forward_backbone(self, inputs: Dict[str, torch.Tensor]) -> Any:
        outputs = self.backbone(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        return outputs

    def _last_hidden_state(self, outputs: Any) -> torch.Tensor:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            language_outputs = getattr(outputs, "language_model_outputs", None)
            hidden_states = getattr(language_outputs, "hidden_states", None)
        return hidden_states[-1] if hidden_states is not None else outputs.last_hidden_state

    def _encode_image_tokens(self, texts: Sequence[str], images_or_paths: Sequence[Any]) -> torch.Tensor:
        inputs = self._processor_inputs(texts, images_or_paths)
        outputs = self._forward_backbone(inputs)
        return self._image_tokens_from_outputs(outputs, inputs)

    def image_forward(self, texts: Sequence[str], images_or_paths: Sequence[Any]) -> torch.Tensor:
        image_tokens = self._encode_image_tokens(texts, images_or_paths)
        pooled = image_tokens.mean(dim=1)
        if self.projector.weight.device != pooled.device:
            self.projector.to(pooled.device)
        return F.normalize(self.projector(pooled.float()), p=2, dim=-1)

    def forward(self, texts: Sequence[str], images_or_paths: Sequence[Any]) -> torch.Tensor:
        return self.image_forward(texts, images_or_paths)

    def _image_tokens_from_outputs(self, outputs: Any, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_tokens = getattr(outputs, "image_hidden_states", None)
        if image_tokens is not None:
            return image_tokens

        hidden = self._last_hidden_state(outputs)
        input_ids = inputs.get("input_ids")
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(self.backbone.config, "image_token_id", None)
        if input_ids is None or image_token_id is None:
            raise ValueError("Cannot find InternVL image tokens for satellite grid pooling.")
        if hasattr(image_token_id, "item"):
            image_token_id = image_token_id.item()

        image_mask = input_ids == int(image_token_id)
        batch_size = int(input_ids.shape[0])
        token_counts = image_mask.sum(dim=1)
        if int(token_counts.min().item()) != int(token_counts.max().item()):
            raise ValueError(f"Uneven image token counts in batch: {token_counts.tolist()}")
        return hidden[image_mask].view(batch_size, int(token_counts[0].item()), hidden.shape[-1])

    def satellite_grid_forward(self, texts: Sequence[str], images: Sequence[Image.Image]) -> torch.Tensor:
        image_tokens = self._encode_image_tokens(texts, images)
        batch_size, token_count, channels = image_tokens.shape
        grid_size = int(token_count**0.5)
        if grid_size * grid_size != token_count:
            raise ValueError(f"Expected square image token grid, got {token_count} tokens.")
        feature_map = image_tokens.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
        pooled = F.adaptive_avg_pool2d(feature_map, (3, 3)).flatten(2).transpose(1, 2)
        if self.projector.weight.device != pooled.device:
            self.projector.to(pooled.device)
        return F.normalize(self.projector(pooled.float()), p=2, dim=-1)


class EncoderNGCGRetrieval(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.encoder = MLLMQueryEncoder(
            model_name=args.mllm_model_name,
            cache_dir=args.cache_dir,
            proj_dim=args.proj_dim,
            dtype=args.dtype,
            load_4bit=args.load_4bit,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            local_files_only=args.local_files_only,
        )

    def forward(self, batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        query_prompts = [IMAGE_PROMPT] * len(batch["query_image"])
        satellite_prompts = [IMAGE_PROMPT] * len(batch["search_image"])
        query_feats = self.encoder.image_forward(query_prompts, batch["query_image"])
        grid_feats = self.encoder.satellite_grid_forward(satellite_prompts, batch["search_image"])
        return query_feats.to(device), grid_feats.to(device)

    def query_forward(self, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
        query_prompts = [IMAGE_PROMPT] * len(batch["query_image"])
        return self.encoder.image_forward(query_prompts, batch["query_image"]).to(device)

    def gallery_forward(
        self,
        images: Sequence[Image.Image],
        device: torch.device,
    ) -> torch.Tensor:
        satellite_prompts = [IMAGE_PROMPT] * len(images)
        return self.encoder.satellite_grid_forward(satellite_prompts, images).to(device)


def move_model_for_training(model: EncoderNGCGRetrieval, device: torch.device) -> EncoderNGCGRetrieval:
    if model.encoder.uses_device_map:
        return model
    return model.to(device)


def create_loader(split: str, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    image_processor = SimpleImageProcessor(size=(256, 256))
    sat_processor = SimpleImageProcessor(size=(int(args.sat_size[0]), int(args.sat_size[1])))
    tokenizer = SimpleTokenizer()

    dataset = ShiftedSatelliteDroneDataset(
        processor=image_processor,
        processor_sat=sat_processor,
        tokenizer=tokenizer,
        split=split,
        subset_heights=args.subset_heights,
        subset_angles=args.subset_angles,
    )
    dataset.return_search_image = True
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate_mllm_batch,
        **build_dataloader_kwargs(args.num_workers, drop_last=shuffle),
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_loader = create_loader("train", args, shuffle=True)
    model = move_model_for_training(EncoderNGCGRetrieval(args), device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found.")
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            query_feats, grid_feats = model(batch, device)
            candidate_feats = grid_feats.reshape(-1, args.proj_dim)
            local_indices = batch["index"].to(device, non_blocking=True)
            satellite_ids = batch.get("satellite_id")
            if satellite_ids is not None:
                satellite_ids = satellite_ids.to(device, non_blocking=True)
            targets = build_retrieval_soft_targets(
                local_indices=local_indices,
                satellite_ids=satellite_ids,
                num_locations=grid_feats.shape[1],
            )
            loss = info_nce_loss(query_feats, candidate_feats, targets, temperature=args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
            optimizer.step()

            total_loss += float(loss.item())
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
        }
        torch.save(checkpoint, Path(args.save_dir) / "last.pth")
        print(f"Epoch {epoch + 1}: loss={checkpoint['train_loss']:.4f}")


@torch.inference_mode()
def extract_features(args: argparse.Namespace) -> FeatureBundle:
    device = torch.device(args.device)
    loader = create_loader("test", args, shuffle=False)
    model = move_model_for_training(EncoderNGCGRetrieval(args), device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state, strict=True)
    model.eval()

    query_feats: List[torch.Tensor] = []
    query_labels: List[str] = []
    query_drone_paths: List[str] = []
    query_satellite_paths: List[str] = []
    query_heights: List[int] = []
    query_angles: List[int] = []
    iou_values: List[float] = []
    center_distances: List[float] = []
    gallery_image_dict: Dict[str, Image.Image] = {}
    gallery_path_dict: Dict[str, str] = {}

    for batch in tqdm(loader, desc="Extract queries [mllm]"):
        query_batch = model.query_forward(batch, device)
        query_batch = F.normalize(query_batch, p=2, dim=1).detach().cpu()
        query_feats.append(query_batch)

        labels = [_path_label(path) for path in batch["satellite_path"]]
        query_labels.extend(labels)
        query_drone_paths.extend([str(path) for path in batch["drone_path"]])
        query_satellite_paths.extend([str(path) for path in batch["satellite_path"]])
        query_heights.extend([int(value) for value in batch["height"].tolist()])
        query_angles.extend([int(value) for value in batch["angle"].tolist()])
        iou_values.extend([0.0] * len(labels))
        center_distances.extend([1e9] * len(labels))

        for idx, label in enumerate(labels):
            if label not in gallery_image_dict:
                gallery_image_dict[label] = batch["search_image"][idx].copy()
                gallery_path_dict[label] = str(batch["satellite_path"][idx])

    gallery_labels = sorted(gallery_image_dict.keys())
    gallery_feats: List[torch.Tensor] = []
    for start in tqdm(range(0, len(gallery_labels), args.batch_size), desc="Extract gallery [mllm]"):
        batch_labels = gallery_labels[start : start + args.batch_size]
        images = [gallery_image_dict[label] for label in batch_labels]
        grid_batch = model.gallery_forward(images, device)
        gallery_feats.append(F.normalize(grid_batch, p=2, dim=2).detach().cpu())

    if not query_feats or not gallery_feats:
        raise RuntimeError("No MLLM query/gallery features were extracted.")

    return FeatureBundle(
        query_feats=torch.cat(query_feats, dim=0),
        query_labels=query_labels,
        query_drone_paths=query_drone_paths,
        query_satellite_paths=query_satellite_paths,
        query_heights=query_heights,
        query_angles=query_angles,
        iou_values=iou_values,
        center_distances=center_distances,
        gallery_labels=gallery_labels,
        gallery_satellite_paths=[gallery_path_dict[label] for label in gallery_labels],
        gallery_feats=torch.cat(gallery_feats, dim=0),
    )


def test(args: argparse.Namespace) -> Dict[str, Any]:
    seed_everything(args.seed)
    bundle = extract_features(args)
    include_map = _load_include_map(args.include_file)
    scored = score_retrieval_and_uiou(
        model_type="encoder_heat",
        bundle=bundle,
        include_map=include_map,
        device=torch.device(args.device),
        candidate_size=args.candidate_size,
        unify_score_mode="global",
        sampling_seed=args.seed,
        encoder_heat_text_score_weight=0.0,
        encoder_heat_text_rerank_topk=0,
    )
    records = scored.pop("query_records")
    per_subset, per_height, per_angle = group_summaries(records)
    overall = summarize_records(records)
    metrics = {
        "model_type": "mllm_ngcg",
        "checkpoint": args.checkpoint or str(Path(args.save_dir) / "last.pth"),
        "sat_size": {"height": int(args.sat_size[0]), "width": int(args.sat_size[1])},
        "candidate_size": args.candidate_size,
        "test_crop_ratio": 1.0,
        "seed": int(args.seed),
        "include_file": args.include_file,
        "subset_heights": args.subset_heights or DEFAULT_SUBSET_HEIGHTS,
        "subset_angles": args.subset_angles or DEFAULT_SUBSET_ANGLES,
        "num_gallery": int(len(bundle.gallery_labels)),
        "num_samples": int(overall["num_samples"]),
        "overall": overall,
        "retrieval": {key: value for key, value in scored.items() if key != "top1_success_drone_names"},
        "top1_success_drone_names": scored["top1_success_drone_names"],
        "per_subset": per_subset,
        "per_height": per_height,
        "per_angle": per_angle,
        "query_records": records if args.save_query_records else [],
        "note": "InternVL MLLM retrieval model; satellite grid is pooled from InternVL image hidden states. Bbox/uIoU fields use zero IoU placeholders.",
    }
    out_file = save_metrics(metrics, args.output_dir, name_suffix=args.output_suffix)
    print_summary(metrics, out_file)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test NGCG-style MLLM retrieval model.")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train")
    parser.add_argument("--mllm-model-name", default=DEFAULT_MLLM_MODEL_NAME)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-suffix", default="mllm_ngcg")
    parser.add_argument("--include-file", default=DEFAULT_INCLUDE_FILE)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--proj-dim", type=int, default=PROJECTION_DIM)
    parser.add_argument("--sat-size", type=int, nargs=2, default=list(SAT_SIZE), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--subset-heights", type=int, nargs="*", default=None)
    parser.add_argument("--subset-angles", type=int, nargs="*", default=None)
    parser.add_argument("--candidate-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,qkv_proj,out_proj",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save-query-records", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_size is not None and args.candidate_size <= 0:
        args.candidate_size = None
    if args.mode in {"train", "train_test"}:
        train(args)
    if args.mode in {"test", "train_test"}:
        if args.checkpoint is None:
            args.checkpoint = str(Path(args.save_dir) / "last.pth")
        test(args)


if __name__ == "__main__":
    main()
