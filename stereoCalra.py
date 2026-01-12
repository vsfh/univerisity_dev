import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
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
DRONE_VIEW_FOLDER = "/data/feihong/drone_view"
IMAGE_FOLDER = "/data/feihong/image_1024"

NUM_EPOCHS = 40
BATCH_SIZE = 5
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class TargetSearchDataset(Dataset):
    def __init__(self, image_pairs, processor, img_to_text_dict, mode='train'):
        # Simplified to directly use the list of pairs
        if mode == 'train':
            self.image_paths = image_pairs[:-400]
        else:
            self.image_paths = image_pairs[-400:]
        self.qprocessor = processor[0]
        self.sprocessor = processor[1]
        self.tokenizer = processor[2]
        self.img_to_text_dict = img_to_text_dict
        
        # This crop size should be based on the desired final image size, not hardcoded if possible
        self.crop_size = 2160 // 5 * 3 # 1296
        self.mode = mode
        self.remove = ['**', '\n', 'noun', 'phrases', 'Phrase', 'Noun', 'Summary', 'Environment', '32 tokens']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        query_path, search_path = self.image_paths[idx]


        if self.mode == 'train':
            drone_dir = os.path.dirname(query_path)
            choice = []
            for number in ['01','21','31','41','51']:
                new_query_path = query_path.replace('01', number)
                if not os.path.exists(new_query_path):
                    continue
                choice.append(new_query_path)
                
            query_path = random.sample(choice, 1)[0]
                
        name = query_path.split('/')[-2]+'_'+query_path.split('/')[-1].split('.')[0][-2:]
        text_description = self.img_to_text_dict[name]

        for noun in self.remove:
            text_description = text_description.replace(noun,'')

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

        text_inputs = self.tokenizer(
            text_description,
            padding="max_length", # Pad to max_length for batching
            truncation=True,      # Truncate long texts
            max_length=64,        # SigLIP typically uses max length 64
            return_tensors="pt"
        )
        return {
            'target_pixel_values': query_inputs['pixel_values'][0],
            'search_pixel_values': search_inputs['pixel_values'][0],
            'input_ids': text_inputs['input_ids'][0],
            'index': index
        }

# --- ARCHITECTURE FIX: Simplified and robust search encoder ---
class Encoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = AutoModel.from_pretrained(model_path)
        self.backbone = model.vision_model
        self.text_encoder = model.text_model

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
    
    def text_forward(self, input_ids):
        """Processes tokenized text and returns the un-normalized feature."""
        
        # 1. Get Text features (last hidden state)
        # The output of the text backbone (e.g., a BERT-like model)
        outputs = self.text_encoder(input_ids=input_ids).pooler_output

        return outputs
    
# --- END ARCHITECTURE FIX ---

def info_nce_loss(query_feats, candidate_feats, positive_indices, temperature=0.07):
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature

    return F.cross_entropy(sim_matrix, positive_indices)

def compute_siglip_loss(query_features, target_features, logit_scale):
    """Computes the pairwise sigmoid loss for contrastive learning."""
    # Compute similarity matrix (logits)
    # Note: Features are typically normalized before loss computation in SigLIP
    query_features = F.normalize(query_features, dim=-1)
    target_features = F.normalize(target_features, dim=-1)
    
    logits = torch.matmul(query_features, target_features.t()) * logit_scale.exp()
    
    # Target: identity matrix (on-diagonal pairs are positive)
    target = torch.eye(logits.shape[0], device=logits.device)
    
    # Sigmoid loss (Binary Cross-Entropy with Logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
    return loss

def compute_image_text_siglip_loss(image_features, text_features):
    """
    Standard SigLIP-style pairwise sigmoid loss for image-text alignment.
    image_features: [B, D]
    text_features: [B, D]
    Returns scalar loss.
    """
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    logits = torch.matmul(image_features, text_features.t())
    target = torch.eye(logits.shape[0], device=logits.device)  # [B, B]
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
    return loss

def cosine_similarity_loss(tensor1, tensor2, positive_pairs=True, margin=0.2):
    """
    Compute cosine similarity loss between two (n, 100, 768) tensors along the last dimension.
    
    Args:
        tensor1 (torch.Tensor): First tensor of shape (n, 100, 768).
        tensor2 (torch.Tensor): Second tensor of shape (n, 100, 768).
        positive_pairs (bool): If True, assumes positive pairs (loss = 1 - similarity).
                             If False, assumes negative pairs (loss = max(0, similarity - margin)).
        margin (float): Margin for negative pairs loss (default: 0.2).
    
    Returns:
        torch.Tensor: Mean cosine similarity loss across the n*100 pairs.
    """
    # Validate input shapes
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
    assert tensor1.shape[-1] == 768, "Last dimension must be 768"
    
    pred_centered = tensor1 - tensor1.mean(dim=-1, keepdim=True)
    target_centered = tensor2 - tensor2.mean(dim=-1, keepdim=True)
    
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(pred_centered, target_centered, dim=-1)
    
    # Compute loss
    if positive_pairs:
        # Positive pairs: loss = 1 - cosine_similarity
        loss = 1.0 - cosine_sim
    else:
        # Negative pairs: loss = max(0, cosine_similarity - margin)
        loss = torch.relu(cosine_sim - margin)
    
    # Return mean loss over all n*100 pairs
    return torch.mean(loss)
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Encoder(MODEL_PATH).to(DEVICE)
    model.load_state_dict(torch.load("../ckpt/train_smooth/best.pth", map_location='cpu'), strict=False)
    model.train()
    optimizer = torch.optim.AdamW(model.text_encoder.parameters(), lr=LEARNING_RATE)

    img_to_text_dict = json.load(open('drone_text_single_long.json', 'r'))
    # txt_json = json.load(open('data_list.json', 'r'))
    # for data in txt_json:
    #     key = data['image_path'][:-4]
    #     text = data['text']
    #     img_to_text_dict[key] = text
        
        
    print("Setting up dataset and dataloader...")
    train_dataset = TargetSearchDataset(image_pairs=image_pairs, processor=(qprocessor, sprocessor, tokenizer), img_to_text_dict=img_to_text_dict)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)

    test_dataset = TargetSearchDataset(image_pairs=image_pairs, processor=(qprocessor, sprocessor, tokenizer), img_to_text_dict=img_to_text_dict, mode='test')
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
            input_ids = batch['input_ids'].to(DEVICE)
            # The local index (0-8) for each item in the batch
            local_indices = batch['index'].to(DEVICE)
            

            # with torch.no_grad():
                # anchor_feats = model.query_forward(target_pixel_values)

            grid_feats = model.ref_forward(search_pixel_values)
            candidate_feats = grid_feats.reshape(-1, model.feature_dim)
            text_feats = model.text_forward(input_ids)
            # img_text_loss = cosine_similarity_loss(text_feats, anchor_feats)

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
            
            img_text_loss = info_nce_loss(text_feats, candidate_feats, positive_indices)
            
            # search_loss = info_nce_loss(anchor_feats, candidate_feats, positive_indices)
            # loss = search_loss + img_text_loss
            loss = img_text_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # progress_bar.set_postfix({'search_loss': f'{search_loss.item():.4f}'})
            progress_bar.set_postfix({'img_text_loss': f'{img_text_loss.item():.4f}'})
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_dataloader) + i)
        avg_loss = total_loss / len(train_dataloader)
            
        model.eval()
        _, txt_val_loss = validation(model, test_dataloader, epoch)
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        # writer.add_scalar('Loss/val_img_epoch', val_loss, epoch)
        writer.add_scalar('Loss/val_txt_epoch', txt_val_loss, epoch)
        print(f"Epoch {epoch+1} finished. Average train Loss: {avg_loss:.4f}. Average test Loss: {txt_val_loss:.4f}")
        
        if txt_val_loss < min_loss:
            torch.save(model.state_dict(), f"{save_path}/best.pth")
            min_loss = txt_val_loss
        else:
            not_update += 1
        if not_update > 5:
            break
    print("Training complete.")

def validation(model, loader, epoch):
    progress_bar = tqdm(loader, desc=f"Valid {epoch+1}/{NUM_EPOCHS}")
    total_loss = 0
    txt_total = 0
    for batch in progress_bar:
        target_pixel_values = batch['target_pixel_values'].to(DEVICE)
        search_pixel_values = batch['search_pixel_values'].to(DEVICE)
        input_ids = batch['input_ids'].to(DEVICE)
        # The local index (0-8) for each item in the batch
        local_indices = batch['index'].to(DEVICE)
        batch_offsets = torch.arange(local_indices.shape[0], device=DEVICE) * 9
        positive_indices = local_indices + batch_offsets
        
        with torch.no_grad():
            anchor_feats = model.query_forward(target_pixel_values)
            grid_feats = model.ref_forward(search_pixel_values)
            text_feats = model.text_forward(input_ids)
            candidate_feats = grid_feats.reshape(-1, model.feature_dim)
            loss = info_nce_loss(anchor_feats, candidate_feats, positive_indices)
            img_txt_loss = info_nce_loss(text_feats, candidate_feats, positive_indices)
            total_loss += loss.item()
            txt_total += img_txt_loss
    avg_loss = total_loss / len(loader)
    txt_avg_loss = txt_total / len(loader)
    return avg_loss, txt_avg_loss
    
def eval(run=False):
    def calcu_cos(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b_norm, a_norm[...,None]).flatten()
    if run:
        img_to_text_dict = json.load(open('drone_text_single_long.json', 'r'))

        qprocessor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        sprocessor = AutoImageProcessor.from_pretrained(MODEL_PATH, do_center_crop=False, do_resize=False)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        encoder = Encoder(MODEL_PATH).to(DEVICE)
        encoder.load_state_dict(torch.load("../ckpt/train_align/best.pth", map_location='cpu'))
        encoder.eval()

        res_search = {}
        res_query = {}
        res_text = {}
        print("Setting up dataset and dataloader...")
        for img_path in tqdm(glob(f'{DRONE_VIEW_FOLDER}/*/*/image-01.jpeg')[-400:]):
            name = img_path.split('/')[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            if os.path.exists(search_path):
                query_image = Image.open(img_path).convert("RGB")
                search_image = Image.open(search_path).convert("RGB")
                search_image = search_image.crop((840,0, 3000, 2160)).resize((640, 640))
                text_description = img_to_text_dict[name+'_01']
                # Further evaluation code would go here
            # query_inputs = qprocessor(images=query_image, return_tensors="pt")['pixel_values'].to(DEVICE)
            search_inputs = sprocessor(images=search_image, return_tensors="pt")['pixel_values'].to(DEVICE)
            input_ids = tokenizer(
                text_description,
                padding="max_length", # Pad to max_length for batching
                truncation=True,      # Truncate long texts
                max_length=64,        # SigLIP typically uses max length 64
                return_tensors="pt"
            )['input_ids'].to(DEVICE)
            with torch.no_grad():
                # anchor_feats = encoder.query_forward(query_inputs)
                # Get the 3x3 grid of features, shape: [B, 9, D]
                grid_feats = encoder.ref_forward(search_inputs)
                text_feats = encoder.text_forward(input_ids)
            res_search[name] = grid_feats.cpu().numpy()
            # res_query[name] = anchor_feats.cpu().numpy()
            res_text[name] = text_feats.cpu().numpy()
        np.savez("eval_search.npz", **res_search)
        np.savez("eval_query.npz", **res_query)
        np.savez("eval_text.npz", **res_text)
        print("Evaluation complete.")
    else:
        search_res = np.load("eval_search.npz")
        query_res = np.load("eval_text.npz")
        distances = json.load(open('distances.json','r'))
        test_num  = 100
        test_list = [k for k in search_res.keys()]
        res = {}
        top1 = 0
        top5 = 0
        top10 = 0
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
                # if distances[f'{name}.kml'][f'{key}.kml']<1000:
                #     candidate.append(i)
            res[key] = []
            for img_name in ex_img_list:
                res[key].append(calcu_cos(drone_feature, search_res[img_name]))
            img_res = np.array(res[key]).mean(1).argsort()[-15:][::-1]
            # if not img_res[0] in candidate:
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
            
        

        
if __name__ == '__main__':
    # exp_name = 'train_satellite'
    # exp_name = 'train_drone'
    # exp_name = 'train_single'
    # exp_name = 'train_smooth'
    # exp_name = 'train_align'
    # if os.path.exists(f'../ckpt/{exp_name}'):
    #     print('other name')
    # else:
    #     os.makedirs(f'../ckpt/{exp_name}')
    #     main(f'../ckpt/{exp_name}')
    # eval(True)
    eval(False)

