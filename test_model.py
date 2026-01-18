import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoTokenizer,
    CLIPProcessor,
    CLIPModel,
)
from PIL import Image
from tqdm import tqdm
import os
from glob import glob
import numpy as np
import random
import json

# --- Configuration ---
CACHE_DIR = "/data/feihong/hf_cache"
DRONE_VIEW_FOLDER = "/data/feihong/drone_view"
IMAGE_FOLDER = "/data/feihong/image_1024"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECTION_DIM = 768


def calcu_cos(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm[..., None]).flatten()


def get_clip_encoder_and_processor():
    MODEL_NAME = "openai/clip-vit-base-patch32"

    class Encoder(nn.Module):
        def __init__(self, model_name, proj_dim=768):
            super().__init__()
            self.clip_model = CLIPModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.vision_model = self.clip_model.vision_model
            self.text_model = self.clip_model.text_model
            self.feature_dim = self.vision_model.config.hidden_size
            self.text_feature_dim = self.text_model.config.hidden_size

            self.pool = nn.AdaptiveAvgPool2d((3, 3))

            self.text_projector = nn.Sequential(
                nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
                nn.ReLU(),
                nn.Linear(self.text_feature_dim * 2, proj_dim),
            )

        def ref_forward(self, pixel_values):
            outputs = self.vision_model(pixel_values)
            patch_tokens = outputs.last_hidden_state
            patch_tokens = patch_tokens[:, :-1, :]

            B, N, D = patch_tokens.shape
            H = W = int(N**0.5)

            patch_tokens_for_pooling = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
            pooled_features = self.pool(patch_tokens_for_pooling)
            final_features = pooled_features.flatten(2).permute(0, 2, 1)

            return final_features

        def query_forward(self, pixel_values):
            pooled_features = self.vision_model(pixel_values).pooler_output
            return pooled_features

        def text_forward(self, input_ids, attention_mask):
            outputs = self.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            pooler_output = outputs.pooler_output
            projected_features = self.text_projector(pooler_output)
            return projected_features

        def forward_fusion(self, vision_feat, text_feat):
            combined_input = torch.cat([vision_feat, text_feat], dim=1)
            fused_feat = self.fusion_mlp(combined_input)
            return fused_feat

    processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    return Encoder, processor, MODEL_NAME


def get_siglip_encoder_and_processor():
    VISION_MODEL_PATH = "/data/feihong/hf_cache/models--google--siglip-base-patch16-224"
    TEXT_MODEL_NAME = "nvidia/NV-Embed-v1"

    class Encoder(nn.Module):
        def __init__(self, vision_model_path, text_model_name, proj_dim=768):
            super().__init__()
            vision_model = AutoModel.from_pretrained(vision_model_path)
            self.backbone = vision_model.vision_model
            self.feature_dim = self.backbone.config.hidden_size

            self.text_encoder = AutoModel.from_pretrained(
                text_model_name, trust_remote_code=True
            )
            self.text_feature_dim = self.text_encoder.config.hidden_size

            self.pool = nn.AdaptiveAvgPool2d((3, 3))

            self.vision_projector = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim * 2),
                nn.ReLU(),
                nn.Linear(self.feature_dim * 2, proj_dim),
            )

            self.text_projector = nn.Sequential(
                nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
                nn.ReLU(),
                nn.Linear(self.text_feature_dim * 2, proj_dim),
            )

            self.fusion_mlp = nn.Sequential(
                nn.Linear(proj_dim * 2, proj_dim * 2),
                nn.ReLU(),
                nn.Linear(proj_dim * 2, proj_dim),
            )

        def ref_forward(self, pixel_values):
            outputs = self.backbone(pixel_values, interpolate_pos_encoding=True)
            patch_tokens = outputs.last_hidden_state

            B, N, D = patch_tokens.shape
            H = W = int(N**0.5)

            patch_tokens_for_pooling = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
            pooled_features = self.pool(patch_tokens_for_pooling)
            final_features = pooled_features.flatten(2).permute(0, 2, 1)

            projected_features = self.vision_projector(final_features)
            return projected_features

        def query_forward(self, pixel_values):
            pooled_features = self.backbone(pixel_values).pooler_output
            projected_features = self.vision_projector(pooled_features)
            return projected_features

        def text_forward(self, input_ids, attention_mask):
            outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            last_hidden_state = outputs.last_hidden_state

            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled_features = sum_embeddings / sum_mask

            projected_features = self.text_projector(mean_pooled_features)
            return projected_features

        def forward_fusion(self, vision_feat, text_feat):
            combined_input = torch.cat([vision_feat, text_feat], dim=1)
            fused_feat = self.fusion_mlp(combined_input)
            return fused_feat

    qprocessor = AutoImageProcessor.from_pretrained(VISION_MODEL_PATH)
    sprocessor = AutoImageProcessor.from_pretrained(
        VISION_MODEL_PATH, do_center_crop=False, do_resize=False
    )
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True)

    return (
        Encoder,
        (qprocessor, sprocessor, tokenizer),
        (VISION_MODEL_PATH, TEXT_MODEL_NAME),
    )


def get_evaclip_encoder_and_processor():
    pass


def get_openclip_encoder_and_processor():
    pass


def extract_features(model_type, encoder_class, processor, model_args, run_flag=True):
    img_to_text_dict = json.load(open("/data/feihong/drone_text_single_long.json", "r"))

    if model_type == "clip":
        encoder = encoder_class(*model_args, proj_dim=PROJECTION_DIM).to(DEVICE)
        model_path = f"../ckpt/train_clip/best.pth"
        npz_prefix = "eval_clip"
    elif model_type == "siglip":
        encoder = encoder_class(*model_args, proj_dim=PROJECTION_DIM).to(DEVICE)
        model_path = f"../ckpt/train_projected_nv_embed/best.pth"
        if not os.path.exists(model_path):
            model_path = f"../ckpt/train_add/best.pth"
        npz_prefix = "eval_siglip"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if os.path.exists(model_path):
        encoder.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model path not found: {model_path}")
        return

    encoder.eval()

    res_search = {}
    res_fused_query = {}

    print("Setting up dataset and dataloader for eval...")
    eval_list = glob(f"{DRONE_VIEW_FOLDER}/*/*/image-01.jpeg")
    if len(eval_list) > 400:
        eval_list = eval_list[-400:]

    remove_terms = [
        "**",
        "\n",
        "noun",
        "phrases",
        "Phrase",
        "Noun",
        "Summary",
        "Environment",
        "32 tokens",
    ]

    for img_path in tqdm(eval_list):
        name = img_path.split("/")[-2]
        search_path = f"{IMAGE_FOLDER}/{name}.png"
        if os.path.exists(search_path):
            try:
                query_image = Image.open(img_path).convert("RGB")
                search_image = Image.open(search_path).convert("RGB")
                search_image = search_image.crop((840, 0, 3000, 2160)).resize(
                    (640, 640)
                )

                text_name = name + "_01"
                text_description = img_to_text_dict.get(text_name, "")
                for term in remove_terms:
                    text_description = text_description.replace(term, "")

                if model_type == "clip":
                    query_inputs = processor(images=query_image, return_tensors="pt")[
                        "pixel_values"
                    ].to(DEVICE)
                    search_inputs = processor(images=search_image, return_tensors="pt")[
                        "pixel_values"
                    ].to(DEVICE)
                    text_inputs = processor(
                        text=text_description,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=77,
                    )
                    input_ids = text_inputs["input_ids"].to(DEVICE)
                    attention_mask = text_inputs["attention_mask"].to(DEVICE)
                elif model_type == "siglip":
                    qprocessor, sprocessor, tokenizer = processor
                    query_inputs = qprocessor(images=query_image, return_tensors="pt")[
                        "pixel_values"
                    ].to(DEVICE)
                    search_inputs = sprocessor(
                        images=search_image, return_tensors="pt"
                    )["pixel_values"].to(DEVICE)
                    text_inputs = tokenizer(
                        text_description,
                        padding="max_length",
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                    )
                    input_ids = text_inputs["input_ids"].to(DEVICE)
                    attention_mask = text_inputs["attention_mask"].to(DEVICE)

                with torch.no_grad():
                    anchor_feats = encoder.query_forward(query_inputs)
                    grid_feats = encoder.ref_forward(search_inputs)
                    text_feats = encoder.text_forward(input_ids, attention_mask)
                    fused_query_feats = encoder.forward_fusion(anchor_feats, text_feats)

                res_search[name] = grid_feats.cpu().numpy()
                res_fused_query[name] = fused_query_feats.cpu().numpy()
            except Exception as e:
                print(f"Error processing {name}: {e}")

    np.savez(f"{npz_prefix}_search.npz", **res_search)
    np.savez(f"{npz_prefix}_fused_query.npz", **res_fused_query)
    print(f"Evaluation feature extraction complete. Saved to {npz_prefix}_*.npz")


def calculate_metrics(model_type):
    if model_type == "clip":
        npz_prefix = "eval_clip"
    elif model_type == "siglip":
        npz_prefix = "eval_siglip"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    search_res = np.load(f"{npz_prefix}_search.npz")
    fused_query_res = np.load(f"{npz_prefix}_fused_query.npz")

    distances = json.load(open("distances.json", "r"))
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
            if distances[f"{name}.kml"][f"{key}.kml"] < 1000:
                candidate_indices.append(i)

        res[key] = []
        for img_name in ex_img_list:
            cos_sim_grid = calcu_cos(fused_query_feature, search_res[img_name])
            res[key].append(cos_sim_grid)

        img_res = np.array(res[key]).mean(1).argsort()[-15:][::-1]

        if any(cand in candidate_indices for cand in img_res[:1]):
            top1 += 1
        if any(cand in candidate_indices for cand in img_res[:5]):
            top5 += 1
        if any(cand in candidate_indices for cand in img_res[:10]):
            top10 += 1

    print(f"\n=== {model_type.upper()} Evaluation Results ===")
    print(f"Top 1: {top1} / {len(test_list)} ({top1 / len(test_list) * 100:.2f}%)")
    print(f"Top 5: {top5} / {len(test_list)} ({top5 / len(test_list) * 100:.2f}%)")
    print(f"Top 10: {top10} / {len(test_list)} ({top10 / len(test_list) * 100:.2f}%)")


def eval_model(model_type, run=False):
    if model_type == "clip":
        encoder_class, processor, model_args = get_clip_encoder_and_processor()
    elif model_type == "siglip":
        encoder_class, processor, model_args = get_siglip_encoder_and_processor()
    else:
        print(f"Model type {model_type} not implemented yet.")
        return

    if run:
        extract_features(
            model_type, encoder_class, processor, model_args, run_flag=True
        )
    else:
        calculate_metrics(model_type)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["clip", "siglip", "evaclip", "openclip"],
        required=True,
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Extract features (run=True) or calculate metrics (run=False)",
    )

    args = parser.parse_args()

    eval_model(args.model, run=args.run)
