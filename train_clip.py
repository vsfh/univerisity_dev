import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm
import os
from glob import glob
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import json

# --- Configuration ---
MODEL_NAME = "openai/clip-vit-base-patch32"
# MODEL_NAME = "openai/clip-vit-large-patch14"
CACHE_DIR = "/data/feihong/hf_cache"
DRONE_VIEW_FOLDER = "/data/feihong/drone_view"
IMAGE_FOLDER = "/data/feihong/image_1024"

NUM_EPOCHS = 40
BATCH_SIZE = 20
LEARNING_RATE = 1e-5
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
PROJECTION_DIM = 768


class TargetSearchDataset(Dataset):
    def __init__(self, image_pairs, processor, img_to_text_dict, mode="train"):
        self.image_paths = image_pairs
        self.processor = processor
        self.img_to_text_dict = img_to_text_dict

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
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        query_path, search_path = self.image_paths[idx]

        if self.mode == "train":
            choice = []
            for number in ["01", "21", "31", "41", "51"]:
                new_query_path = query_path.replace("01", number)
                if not os.path.exists(new_query_path):
                    continue
                choice.append(new_query_path)

            if choice:
                query_path = random.sample(choice, 1)[0]

        name = (
            query_path.split("/")[-2]
            + "_"
            + query_path.split("/")[-1].split(".")[0][-2:]
        )
        text_description = self.img_to_text_dict.get(name, "")

        for noun in self.remove:
            text_description = text_description.replace(noun, "")

        try:
            query_image = Image.open(query_path).convert("RGB")
            search_image = Image.open(search_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping this item.")
            return self.__getitem__((idx + 1) % len(self))

        if self.mode == "train":
            width, height = search_image.size

            min_left = (width / 2) - self.crop_size
            max_left = width / 2
            min_top = (height / 2) - self.crop_size
            max_top = height / 2

            min_left = max(0, min_left)
            min_top = max(0, min_top)
            max_left = min(width - self.crop_size, max_left)
            max_top = min(height - self.crop_size, max_top)

            if max_left < min_left:
                max_left = min_left
            if max_top < min_top:
                max_top = min_top

            left = random.uniform(min_left, max_left)
            top = random.uniform(min_top, max_top)

            right = left + self.crop_size
            bottom = top + self.crop_size

            augmented_crop_image = search_image.crop((left, top, right, bottom))
            augmented_crop_image = augmented_crop_image.resize((640, 640))

            center_x_in_crop = (width / 2) - left
            center_y_in_crop = (height / 2) - top

            normalized_center_pos = torch.tensor(
                [center_x_in_crop / self.crop_size, center_y_in_crop / self.crop_size],
                dtype=torch.float32,
            )

            col_idx = min(int(normalized_center_pos[0] * 3), 2)
            row_idx = min(int(normalized_center_pos[1] * 3), 2)
            index = row_idx * 3 + col_idx
        else:
            augmented_crop_image = search_image.crop((840, 0, 3000, 2160)).resize(
                (640, 640)
            )
            index = 4

        # Process images and text with CLIP processor
        query_inputs = self.processor(images=query_image, return_tensors="pt")
        search_inputs = self.processor(images=augmented_crop_image, return_tensors="pt")
        text_inputs = self.processor(
            text=text_description,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )

        return {
            "target_pixel_values": query_inputs["pixel_values"][0],
            "search_pixel_values": search_inputs["pixel_values"][0],
            "input_ids": text_inputs["input_ids"][0],
            "attention_mask": text_inputs["attention_mask"][0],
            "index": index,
        }


class Encoder(nn.Module):
    def __init__(self, model_name, proj_dim=768):
        super().__init__()

        # Load CLIP model
        try:
            self.clip_model = CLIPModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.vision_model = self.clip_model.vision_model
            self.text_model = self.clip_model.text_model
            self.feature_dim = self.vision_model.config.hidden_size
            self.text_feature_dim = self.text_model.config.hidden_size
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise

        # Vision Pooling
        self.pool = nn.AdaptiveAvgPool2d((3, 3))

        # Projection Heads
        # self.vision_projector = nn.Sequential(
        #     nn.Linear(self.feature_dim, self.feature_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(self.feature_dim * 2, proj_dim)
        # )

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.text_feature_dim * 2, proj_dim),
        )

        # Fusion MLP
        # self.fusion_mlp = nn.Sequential(
        #     nn.Linear(proj_dim * 2, proj_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(proj_dim * 2, proj_dim)
        # )

    def ref_forward(self, pixel_values):
        outputs = self.vision_model(pixel_values)
        patch_tokens = outputs.last_hidden_state
        patch_tokens = patch_tokens[:, :-1, :]

        B, N, D = patch_tokens.shape
        H = W = int(N**0.5)

        patch_tokens_for_pooling = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
        pooled_features = self.pool(patch_tokens_for_pooling)
        final_features = pooled_features.flatten(2).permute(0, 2, 1)

        # projected_features = self.vision_projector(final_features)
        return final_features  # [B, 9, 768]

    def query_forward(self, pixel_values):
        pooled_features = self.vision_model(pixel_values).pooler_output
        # projected_features = self.vision_projector(pooled_features)
        return pooled_features  # [B, 768]

    def text_forward(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output

        # mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        # sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        # mean_pooled_features = sum_embeddings / sum_mask

        projected_features = self.text_projector(pooler_output)
        return projected_features  # [B, 768]

    # def forward_fusion(self, vision_feat, text_feat):
    #     combined_input = torch.cat([vision_feat, text_feat], dim=1)
    #     fused_feat = self.fusion_mlp(combined_input)
    #     return fused_feat


def info_nce_loss(query_feats, candidate_feats, positive_indices, temperature=0.07):
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature
    return F.cross_entropy(sim_matrix, positive_indices)


def main(save_path):
    not_update = 0
    exp_name = save_path.split("/")[-1] if save_path else "default_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Gathering image pairs...")
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

    print("Loading models and processor...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    model = Encoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    img_to_text_dict = json.load(open("/data/feihong/drone_text_single_long.json", "r"))

    print("Setting up dataset and dataloader...")
    train_dataset = TargetSearchDataset(
        image_pairs=train_image_pairs,
        processor=processor,
        img_to_text_dict=img_to_text_dict,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4
    )

    test_dataset = TargetSearchDataset(
        image_pairs=test_image_pairs,
        processor=processor,
        img_to_text_dict=img_to_text_dict,
        mode="test",
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4
    )

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")
    min_loss = 100
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for i, batch in enumerate(progress_bar):
            target_pixel_values = batch["target_pixel_values"].to(DEVICE)
            search_pixel_values = batch["search_pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            local_indices = batch["index"].to(DEVICE)

            anchor_feats = model.query_forward(target_pixel_values)
            grid_feats = model.ref_forward(search_pixel_values)
            candidate_feats = grid_feats.reshape(-1, PROJECTION_DIM)
            text_feats = model.text_forward(input_ids, attention_mask)

            positive_indices = torch.zeros(
                local_indices.shape[0], local_indices.shape[0] * 9, device=DEVICE
            )
            batch_offsets = torch.arange(local_indices.shape[0], device=DEVICE) * 9
            row_indices_broad = torch.arange(
                local_indices.shape[0], device=DEVICE
            ).unsqueeze(1)
            col_offsets = torch.arange(9, device=DEVICE).unsqueeze(0)
            same_image_cols = batch_offsets.unsqueeze(1) + col_offsets

            positive_indices[row_indices_broad, same_image_cols] = 0.5
            global_positive_indices = local_indices + batch_offsets
            row_indices_flat = torch.arange(local_indices.shape[0], device=DEVICE)
            positive_indices[row_indices_flat, global_positive_indices] = 0.95

            # combined_query_feats = model.forward_fusion(anchor_feats, text_feats)
            # img_text_loss = info_nce_loss(combined_query_feats, candidate_feats, positive_indices)
            img_loss = info_nce_loss(anchor_feats, candidate_feats, positive_indices)
            text_loss = info_nce_loss(text_feats, candidate_feats, positive_indices)
            loss = img_loss + text_loss

            # loss = img_text_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"img_text_loss": f"{loss.item():.4f}"})
            writer.add_scalar(
                "Loss/train_batch", loss.item(), epoch * len(train_dataloader) + i
            )

        avg_loss = total_loss / len(train_dataloader)

        model.eval()
        val_loss, txt_val_loss = validation(model, test_dataloader, epoch)
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Loss/val_img_epoch", val_loss, epoch)
        writer.add_scalar("Loss/val_txt_epoch", txt_val_loss, epoch)
        print(
            f"Epoch {epoch + 1} finished. Average train Loss: {avg_loss:.4f}. Average test Loss: {txt_val_loss:.4f}"
        )

        if txt_val_loss < min_loss:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/best.pth")
            min_loss = txt_val_loss
            not_update = 0
        else:
            not_update += 1
        if not_update > 5:
            print("Validation loss not improving. Stopping early.")
            break

    print("Training complete.")
    writer.close()


def validation(model, loader, epoch):
    progress_bar = tqdm(loader, desc=f"Valid {epoch + 1}/{NUM_EPOCHS}")
    total_loss = 0
    txt_total = 0
    for batch in progress_bar:
        target_pixel_values = batch["target_pixel_values"].to(DEVICE)
        search_pixel_values = batch["search_pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        local_indices = batch["index"].to(DEVICE)

        batch_offsets = torch.arange(local_indices.shape[0], device=DEVICE) * 9
        positive_indices = local_indices + batch_offsets

        with torch.no_grad():
            anchor_feats = model.query_forward(target_pixel_values)
            grid_feats = model.ref_forward(search_pixel_values)
            text_feats = model.text_forward(input_ids, attention_mask)

            candidate_feats = grid_feats.reshape(-1, PROJECTION_DIM)

            img_txt_loss = info_nce_loss(
                text_feats + anchor_feats, candidate_feats, positive_indices
            )

            total_loss += img_txt_loss.item()
            txt_total += img_txt_loss.item()

    avg_loss = total_loss / len(loader)
    txt_avg_loss = txt_total / len(loader)
    return avg_loss, txt_avg_loss


def eval(run=False):
    def calcu_cos(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b_norm, a_norm[..., None]).flatten()

    if run:
        img_to_text_dict = json.load(
            open("/data/feihong/drone_text_single_long.json", "r")
        )
        processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

        encoder = Encoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
        model_path = f"../ckpt/train_clip/best.pth"
        if os.path.exists(model_path):
            encoder.load_state_dict(torch.load(model_path, map_location="cpu"))
        encoder.eval()

        res_search = {}
        res_fused_query = {}

        print("Setting up dataset and dataloader for eval...")
        eval_list = []
        with open("/data/feihong/ckpt/test.txt", "r") as f:
            for line in f:
                query_path = line.strip()
                eval_list.append(query_path)

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
                    ]:
                        text_description = text_description.replace(noun, "")

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

                    with torch.no_grad():
                        anchor_feats = encoder.query_forward(query_inputs)
                        grid_feats = encoder.ref_forward(search_inputs)
                        text_feats = encoder.text_forward(input_ids, attention_mask)
                        fused_query_feats = anchor_feats + text_feats

                    res_search[name] = grid_feats.cpu().numpy()
                    res_fused_query[name] = fused_query_feats.cpu().numpy()
                except Exception as e:
                    print(f"Error processing {name}: {e}")

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


def eval_denseuav(run=False):
    DENSE_UAV_ROOT = "/data/feihong/DenseUAV"
    TEST_DENSE_FILE = "/data/feihong/ckpt/test_dense.txt"

    def calcu_cos(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b_norm, a_norm[..., None]).flatten()

    if run:
        img_to_text_dict = json.load(
            open("/data/feihong/drone_text_single_long.json", "r")
        )
        processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

        encoder = Encoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
        model_path = f"../ckpt/train_clip/best.pth"
        if os.path.exists(model_path):
            encoder.load_state_dict(torch.load(model_path, map_location="cpu"))
        encoder.eval()

        res_search = {}
        res_fused_query = {}

        print("Setting up dataset and dataloader for eval_denseuav...")
        eval_list = []
        with open(TEST_DENSE_FILE, "r") as f:
            for line in f:
                query_path = line.strip()
                eval_list.append(query_path)

        for img_path in tqdm(eval_list):
            parts = img_path.split("/")
            drone_id = parts[-2]
            is_train = "train" in parts

            if is_train:
                search_path = f"{DENSE_UAV_ROOT}/train/satellite/{drone_id}/H100.tif"
            else:
                search_path = (
                    f"{DENSE_UAV_ROOT}/test/gallery_satellite/{drone_id}/H100.tif"
                )

            if os.path.exists(search_path):
                try:
                    query_image = Image.open(img_path).convert("RGB")
                    search_image = Image.open(search_path).convert("RGB")
                    search_image = search_image.resize((640, 640))

                    text_name = drone_id + "_01"
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
                    ]:
                        text_description = text_description.replace(noun, "")

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

                    with torch.no_grad():
                        anchor_feats = encoder.query_forward(query_inputs)
                        grid_feats = encoder.ref_forward(search_inputs)
                        fused_query_feats = anchor_feats

                    res_search[drone_id] = grid_feats.cpu().numpy()
                    res_fused_query[drone_id] = fused_query_feats.cpu().numpy()
                except Exception as e:
                    print(f"Error processing {drone_id}: {e}")

        np.savez("eval_denseuav_search.npz", **res_search)
        np.savez("eval_denseuav_fused_query.npz", **res_fused_query)
        print("Evaluation feature extraction complete.")
    else:
        search_res = np.load("eval_denseuav_search.npz")
        fused_query_res = np.load("eval_denseuav_fused_query.npz")

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

            ex_img_list = random.sample(test_list, min(99, len(test_list) - 1))
            if key in ex_img_list:
                ex_img_list.remove(key)
            ex_img_list.append(key)
            right_num = len(ex_img_list) - 1

            res[key] = []
            for img_name in ex_img_list:
                cos_sim_grid = calcu_cos(fused_query_feature, search_res[img_name])
                res[key].append(cos_sim_grid)

            img_res = np.array(res[key]).mean(1).argsort()[-15:][::-1]

            if right_num in img_res[:1]:
                top1 += 1
            if right_num in img_res[:5]:
                top5 += 1
            if right_num in img_res[:10]:
                top10 += 1

        print(f"Top 1: {top1} / {len(test_list)} ({top1 / len(test_list) * 100:.2f}%)")
        print(f"Top 5: {top5} / {len(test_list)} ({top5 / len(test_list) * 100:.2f}%)")
        print(
            f"Top 10: {top10} / {len(test_list)} ({top10 / len(test_list) * 100:.2f}%)"
        )


if __name__ == "__main__":
    exp_name = "train_clip"
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
    # eval(True)  # Extract features
    # eval(False)  # Calculate metrics
    eval_denseuav(True)  # Extract features
    eval_denseuav(False)  # Extract features
