import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from tqdm import tqdm
import os
from glob import glob
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import json
# --- Configuration ---
# It's good practice to use variables for paths for easier modification
MODEL_PATH = "/home/SATA4T/gregory/hf_cache/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
DRONE_VIEW_FOLDER = "/home/SATA4T/gregory/data/drone_view"
IMAGE_FOLDER = "/home/SATA4T/gregory/data/image_1024"

NUM_EPOCHS = 40
BATCH_SIZE = 10
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class TargetSearchDataset(Dataset):
    def __init__(self, image_pairs, processor, mode='train'):
        # Simplified to directly use the list of pairs
        if mode == 'train':
            self.image_paths = image_pairs[:-400]
        else:
            self.image_paths = image_pairs[-400:]
        self.qprocessor = processor[0]
        self.sprocessor = processor[1]
        
        # This crop size should be based on the desired final image size, not hardcoded if possible
        self.crop_size = 2160 // 5 * 3 # 1296
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        query_path, search_path = self.image_paths[idx]
        if self.mode == 'train':
            drone_dir = os.path.dirname(query_path)
            query_path = random.sample([os.path.join(drone_dir, image_name) for image_name in os.listdir(drone_dir)], 1)[0]
        try:
            query_image = Image.open(query_path).convert("RGB")
            search_image = Image.open(search_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping this item.")
            return self.__getitem__((idx + 1) % len(self))
        
        if self.mode == 'train':
            width, height = search_image.size
            
            min_left = (width / 2) - self.crop_size
            max_left = width / 2
            min_top = (height / 2) - self.crop_size
            max_top = height / 2

            min_left = max(0, min_left)
            min_top = max(0, min_top)
            max_left = min(width - self.crop_size, max_left)
            max_top = min(height - self.crop_size, max_top)
            
            left = random.uniform(min_left, max_left)
            top = random.uniform(min_top, max_top)

            right = left + self.crop_size
            bottom = top + self.crop_size
            
            augmented_crop_image = search_image.crop((left, top, right, bottom))

            # The search processor expects a certain size, so we resize the crop
            augmented_crop_image = augmented_crop_image.resize((640, 640))
            
            center_x_in_crop = (width / 2) - left
            center_y_in_crop = (height / 2) - top
            
            normalized_center_pos = torch.tensor(
                [center_x_in_crop / self.crop_size, center_y_in_crop / self.crop_size],
                dtype=torch.float32
            )
            
            # --- BUG FIX: Robust index calculation ---
            # Map a coordinate from [0, 1] to an integer index {0, 1, 2}
            # We clamp the result to ensure it's never out of bounds (e.g., if pos is exactly 1.0)
            col_idx = min(int(normalized_center_pos[0] * 3), 2)
            row_idx = min(int(normalized_center_pos[1] * 3), 2)
            index = row_idx * 3 + col_idx
        else:
            augmented_crop_image = search_image.crop((840,0, 3000, 2160)).resize((640, 640))
            # --- END BUG FIX ---
            index = 4
        
        query_inputs = self.qprocessor(images=query_image, return_tensors="pt")
        search_inputs = self.sprocessor(images=augmented_crop_image, return_tensors="pt")

        return {
            'target_pixel_values': query_inputs['pixel_values'][0],
            'search_pixel_values': search_inputs['pixel_values'][0],
            'index': index
        }

# --- ARCHITECTURE FIX: Simplified and robust search encoder ---
class Encoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path).vision_model
        # Use AdaptiveAvgPool2d to pool patch features into a 3x3 grid
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        # The feature dimension of SigLIP patches
        self.feature_dim = self.backbone.config.hidden_size

    def ref_forward(self, pixel_values):
        # Get all patch features from the backbone
        outputs = self.backbone(pixel_values, interpolate_pos_encoding=True)
        # Exclude the [CLS] token, shape: [B, num_patches, D]
        patch_tokens = outputs.last_hidden_state
        
        B, N, D = patch_tokens.shape
        # Vision Transformers have a square root number of patches
        H = W = int(N**0.5)
        
        # Reshape for pooling: [B, D, H, W]
        patch_tokens_for_pooling = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
        
        # Pool the features: [B, D, 3, 3]
        pooled_features = self.pool(patch_tokens_for_pooling)
        
        # Reshape back to the desired output: [B, 9, D]
        final_features = pooled_features.flatten(2).permute(0, 2, 1)
        
        return final_features
    
    def query_forward(self, pixel_values):
        pooled_features = self.backbone(pixel_values).pooler_output

        return pooled_features
    
# --- END ARCHITECTURE FIX ---

def info_nce_loss(query_feats, candidate_feats, positive_indices, temperature=0.07):
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature

    return F.cross_entropy(sim_matrix, positive_indices)

def main(save_path, label_smooth=True):
    not_update = 0
    writer = SummaryWriter(f'runs/{exp_name}')
    print("Gathering image pairs...")
    image_pairs = []
    for query_path in tqdm(glob(f'{DRONE_VIEW_FOLDER}/*/*/image-01.jpeg')):
        name = query_path.split('/')[-2]
        search_path = f"{IMAGE_FOLDER}/{name}.png"
        if os.path.exists(search_path): # Check if the corresponding search image exists
            image_pairs.append((query_path, search_path))

    print(f"Found {len(image_pairs)} valid pairs.")

    print("Loading models and processor...")
    qprocessor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    sprocessor = AutoImageProcessor.from_pretrained(MODEL_PATH, do_center_crop=False, do_resize=False)
    
    model = Encoder(MODEL_PATH).to(DEVICE)
    

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


    print("Setting up dataset and dataloader...")
    train_dataset = TargetSearchDataset(image_pairs=image_pairs, processor=(qprocessor, sprocessor))
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)

    test_dataset = TargetSearchDataset(image_pairs=image_pairs, processor=(qprocessor, sprocessor), mode='test')
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4)
    

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")
    min_loss = 100
    for epoch in range(NUM_EPOCHS):

        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for i, batch in enumerate(progress_bar):
            target_pixel_values = batch['target_pixel_values'].to(DEVICE)
            search_pixel_values = batch['search_pixel_values'].to(DEVICE)
            
            # The local index (0-8) for each item in the batch
            local_indices = batch['index'].to(DEVICE)
            

            anchor_feats = model.query_forward(target_pixel_values)
            grid_feats = model.ref_forward(search_pixel_values)

            candidate_feats = grid_feats.reshape(-1, model.feature_dim)

            if not label_smooth:
                # Calculate the global indices for the positive pairs in the flattened list
                batch_offsets = torch.arange(local_indices.shape[0], device=DEVICE) * 9
                positive_indices = (local_indices + batch_offsets).long()
            else:
                # Start with 0 for all batch negatives
                positive_indices = torch.zeros(local_indices.shape[0], local_indices.shape[0]*9, device=DEVICE)

                # Get indices for all "same-image" candidates (9 per query)
                batch_offsets = torch.arange(local_indices.shape[0], device=DEVICE) * 9
                row_indices_broad = torch.arange(local_indices.shape[0], device=DEVICE).unsqueeze(1) # [B, 1]
                col_offsets = torch.arange(9, device=DEVICE).unsqueeze(0)      # [1, 9]
                same_image_cols = batch_offsets.unsqueeze(1) + col_offsets      # [B, 9]

                # 1. Set all 9 "same-image" regions (hard negatives) to 0.5
                positive_indices[row_indices_broad, same_image_cols] = 0.5

                # 2. Set the single "true-positive" region to 0.95
                global_positive_indices = local_indices + batch_offsets # [B]
                row_indices_flat = torch.arange(local_indices.shape[0], device=DEVICE)       # [B]
                positive_indices[row_indices_flat, global_positive_indices] = 0.95
            
            loss = info_nce_loss(anchor_feats, candidate_feats, positive_indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_dataloader) + i)
        avg_loss = total_loss / len(train_dataloader)
            
        model.eval()
        val_loss = validation(model, test_dataloader, epoch)
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Loss/validation_epoch', val_loss, epoch)
        print(f"Epoch {epoch+1} finished. Average train Loss: {avg_loss:.4f}. Average test Loss: {val_loss:.4f}")
        
        if val_loss < min_loss:
            torch.save(model.state_dict(), f"{save_path}/best.pth")
            min_loss = val_loss
        else:
            not_update += 1
        if not_update > 5:
            break
    print("Training complete.")

def validation(model, loader, epoch):
    progress_bar = tqdm(loader, desc=f"Valid {epoch+1}/{NUM_EPOCHS}")
    total_loss = 0
    for batch in progress_bar:
        target_pixel_values = batch['target_pixel_values'].to(DEVICE)
        search_pixel_values = batch['search_pixel_values'].to(DEVICE)
        
        # The local index (0-8) for each item in the batch
        local_indices = batch['index'].to(DEVICE)
        batch_offsets = torch.arange(local_indices.shape[0], device=DEVICE) * 9
        positive_indices = local_indices + batch_offsets
        
        with torch.no_grad():
            anchor_feats = model.query_forward(target_pixel_values)
            grid_feats = model.ref_forward(search_pixel_values)
            candidate_feats = grid_feats.reshape(-1, model.feature_dim)
            loss = info_nce_loss(anchor_feats, candidate_feats, positive_indices)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss
    
def eval(run=False):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    def calcu_cos(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b_norm, a_norm[...,None]).flatten()
    if run:
        qprocessor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        sprocessor = AutoImageProcessor.from_pretrained(MODEL_PATH, do_center_crop=False, do_resize=False)

        encoder = Encoder(MODEL_PATH).to(DEVICE)
        encoder.load_state_dict(torch.load("../ckpt/train_smooth/best.pth", map_location='cpu'))
        encoder.eval()

        res_search = {}
        res_query = {}
        print("Setting up dataset and dataloader...")
        for img_path in tqdm(glob(f'{DRONE_VIEW_FOLDER}/*/*/image-01.jpeg')[-400:]):
            name = img_path.split('/')[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            if os.path.exists(search_path):
                query_image = Image.open(img_path).convert("RGB")
                search_image = Image.open(search_path).convert("RGB")
                search_image = search_image.crop((840,0, 3000, 2160)).resize((640, 640))
                # Further evaluation code would go here
            query_inputs = qprocessor(images=query_image, return_tensors="pt")['pixel_values'].to(DEVICE)
            search_inputs = sprocessor(images=search_image, return_tensors="pt")['pixel_values'].to(DEVICE)
            with torch.no_grad():
                anchor_feats = encoder.query_forward(query_inputs)
                # Get the 3x3 grid of features, shape: [B, 9, D]
                grid_feats = encoder.ref_forward(search_inputs)
            res_search[name] = grid_feats.cpu().numpy()
            res_query[name] = anchor_feats.cpu().numpy()
        np.savez("eval_search_smooth.npz", **res_search)
        np.savez("eval_query_smooth.npz", **res_query)
        print("Evaluation complete.")
    else:
        search_res = np.load("eval_search_smooth.npz")
        query_res = np.load("eval_query_smooth.npz")
        distances = json.load(open('distances.json','r'))
        test_num  = 100
        test_list = [k for k in search_res.keys()]
        res = {}
        top1 = 0
        top5 = 0
        top10 = 0
        error_num = 0
        image_name_list = [k for k in query_res.keys()]
        for key in tqdm(test_list):
            drone_feature = query_res[key]
            ex_img_list = random.sample(test_list, test_num-1)
            if key in ex_img_list:
                ex_img_list.remove(key)
                ex_img_list.append(image_name_list[-1])
            ex_img_list.append(key)
            candidate = [99]
            for i, name in enumerate(ex_img_list[:-1]):
                if f'{name}.kml' not in distances:
                    continue
                elif f'{key}.kml' not in distances[f'{name}.kml']:
                    continue
                if distances[f'{name}.kml'][f'{key}.kml']<1000:
                    candidate.append(i)
            res[key] = []
            for img_name in ex_img_list:
                res[key].append(calcu_cos(drone_feature, search_res[img_name]))
            img_res = np.array(res[key]).mean(1).argsort()[-15:][::-1]
            
            # if not img_res[0] in candidate:
            #     error_num += 1
            #     print(key, ex_img_list[img_res[0]])
            for cand in img_res[:1]:
                if cand in candidate:
                    top1 += 1
                    break
            for cand in img_res[:5]:
                if cand in candidate:
                    top5 += 1
                    break
            for cand in img_res[:10]:
                if cand in candidate:
                    top10 += 1
                    break
        print(top1, top5, top10)
        print(error_num)
            
        

        
if __name__ == '__main__':
    # exp_name = 'train_satellite'
    # exp_name = 'train_drone'
    # exp_name = 'train_single'
    # exp_name = 'train_smooth'
    # if os.path.exists(f'../ckpt/{exp_name}'):
    #     print('other name')
    # else:
    #     os.makedirs(f'../ckpt/{exp_name}')
    #     main(f'../ckpt/{exp_name}')
    eval(True)
    eval(False)

