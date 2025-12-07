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
# Updated paths to separate vision and text models
VISION_MODEL_PATH = "/home/SATA4T/gregory/hf_cache/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
TEXT_MODEL_NAME = "nvidia/NV-Embed-v1" # Using NV-Embed-v1 as the more powerful LLM
DRONE_VIEW_FOLDER = "/home/SATA4T/gregory/data/drone_view"
IMAGE_FOLDER = "/home/SATA4T/gregory/data/image_1024"

NUM_EPOCHS = 40
BATCH_SIZE = 10
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECTION_DIM = 768 # The target dimension you requested


class TargetSearchDataset(Dataset):
    def __init__(self, image_pairs, processor, img_to_text_dict, mode='train'):
        # ... existing code ...
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
            # ... existing code ...
            choice = []
            for number in ['01','21','31','41','51']:
                new_query_path = query_path.replace('01', number)
                if not os.path.exists(new_query_path):
                    continue
                choice.append(new_query_path)
                
            if choice: # Ensure choice is not empty
                query_path = random.sample(choice, 1)[0]
            # else: remain as original query_path
                
        name = query_path.split('/')[-2]+'_'+query_path.split('/')[-1].split('.')[0][-2:]
        text_description = self.img_to_text_dict.get(name, "") # Use .get for safety

        for noun in self.remove:
            text_description = text_description.replace(noun,'')

        try:
            query_image = Image.open(query_path).convert("RGB")
            search_image = Image.open(search_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping this item.")
            # Recursively get next item
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
            
            # Handle potential case where window is smaller than crop_size
            if max_left < min_left: max_left = min_left
            if max_top < min_top: max_top = min_top
            
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
            
            col_idx = min(int(normalized_center_pos[0] * 3), 2)
            row_idx = min(int(normalized_center_pos[1] * 3), 2)
            index = row_idx * 3 + col_idx
        else:
            augmented_crop_image = search_image.crop((840,0, 3000, 2160)).resize((640, 640))
            index = 4
        
        query_inputs = self.qprocessor(images=query_image, return_tensors="pt")
        search_inputs = self.sprocessor(images=augmented_crop_image, return_tensors="pt")

        text_inputs = self.tokenizer(
            text_description,
            padding="max_length", 
            truncation=True,     
            max_length=128, # Increased max_length
            return_tensors="pt"
        )
        return {
            'target_pixel_values': query_inputs['pixel_values'][0],
            'search_pixel_values': search_inputs['pixel_values'][0],
            'input_ids': text_inputs['input_ids'][0],
            'attention_mask': text_inputs['attention_mask'][0], # Add attention mask
            'index': index
        }

# --- NEW: Encoder with NV-Embed and Fusion MLP ---
class Encoder(nn.Module):
    def __init__(self, vision_model_path, text_model_name, proj_dim=768):
        super().__init__()
        
        # 1. Load Vision Model (SigLIP)
        try:
            vision_model = AutoModel.from_pretrained(vision_model_path)
            self.backbone = vision_model.vision_model
            self.feature_dim = self.backbone.config.hidden_size # Original vision dim (e.g., 768)
        except Exception as e:
            print(f"Error loading vision model from {vision_model_path}: {e}")
            raise
            
        # 2. Load Text Model (e.g., NV-Embed-v1)
        try:
            self.text_encoder = AutoModel.from_pretrained(
                text_model_name, 
                trust_remote_code=True # CRITICAL for this model
            )
            self.text_feature_dim = self.text_encoder.config.hidden_size # Original text dim (e.g., 4096)
        except Exception as e:
            print(f"Error loading text model {text_model_name}: {e}")
            raise

        # 3. Vision Pooling
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        
        # 4. Define MLP Projection Heads
        # Project Vision features down to proj_dim
        self.vision_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, proj_dim)
        )
        
        # Project Text features down to proj_dim
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.text_feature_dim * 2, proj_dim)
        )
        
        # 5. Define Fusion MLP
        # Takes concatenated (vision_proj + text_proj) features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim * 2), # Input: 768*2 = 1536
            nn.ReLU(),
            nn.Linear(proj_dim * 2, proj_dim)     # Output: 768
        )

    def ref_forward(self, pixel_values):
        # Get all patch features from the backbone
        outputs = self.backbone(pixel_values, interpolate_pos_encoding=True)
        patch_tokens = outputs.last_hidden_state
        
        B, N, D = patch_tokens.shape
        H = W = int(N**0.5)
        
        # Reshape for pooling: [B, D, H, W]
        patch_tokens_for_pooling = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
        
        # Pool the features: [B, D, 3, 3]
        pooled_features = self.pool(patch_tokens_for_pooling)
        
        # Reshape: [B, 9, D]
        final_features = pooled_features.flatten(2).permute(0, 2, 1)
        
        # Project vision features
        projected_features = self.vision_projector(final_features)
        
        return projected_features # [B, 9, 768]
    
    def query_forward(self, pixel_values):
        # Get pooled [CLS] token output
        pooled_features = self.backbone(pixel_values).pooler_output # [B, D_vision]
        
        # Project vision features
        projected_features = self.vision_projector(pooled_features)
        
        return projected_features # [B, 768]
    
    def text_forward(self, input_ids, attention_mask):
        """
        Processes tokenized text using hidden states (masked mean pooling)
        and returns the projected feature.
        """
        
        # 1. Get Text features (last hidden state)
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state # [B, SeqLen, Dim]
        
        # 2. Perform masked mean pooling
        # Expand attention_mask to match hidden_state dims: [B, SeqLen] -> [B, SeqLen, 1]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # Sum unmasked token embeddings
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        # Count unmasked tokens
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        # Calculate mean
        mean_pooled_features = sum_embeddings / sum_mask # [B, Dim]

        # 3. Project text features
        projected_features = self.text_projector(mean_pooled_features)
        
        return projected_features # [B, 768]

    def forward_fusion(self, vision_feat, text_feat):
        """Fuses projected vision and text features using an MLP."""
        combined_input = torch.cat([vision_feat, text_feat], dim=1) # [B, 768+768]
        fused_feat = self.fusion_mlp(combined_input)
        return fused_feat
# --- END NEW Encoder ---


def info_nce_loss(query_feats, candidate_feats, positive_indices, temperature=0.07):
    # positive_indices can now be a float tensor for label smoothing
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature

    # Use CrossEntropyLoss for soft labels
    return F.cross_entropy(sim_matrix, positive_indices)

# ... (other loss functions are unchanged) ...
def compute_siglip_loss(query_features, target_features, logit_scale):
    # ... existing code ...
    query_features = F.normalize(query_features, dim=-1)
    target_features = F.normalize(target_features, dim=-1)
    logits = torch.matmul(query_features, target_features.t()) * logit_scale.exp()
    target = torch.eye(logits.shape[0], device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
    return loss

def compute_image_text_siglip_loss(image_features, text_features):
    # ... existing code ...
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = torch.matmul(image_features, text_features.t())
    target = torch.eye(logits.shape[0], device=logits.device)  # [B, B]
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
    return loss

def cosine_similarity_loss(tensor1, tensor2, positive_pairs=True, margin=0.2):
    # ... existing code ...
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
    # Last dim is now PROJECTION_DIM
    assert tensor1.shape[-1] == PROJECTION_DIM, f"Last dimension must be {PROJECTION_DIM}"
    
    pred_centered = tensor1 - tensor1.mean(dim=-1, keepdim=True)
    target_centered = tensor2 - tensor2.mean(dim=-1, keepdim=True)
    
    cosine_sim = F.cosine_similarity(pred_centered, target_centered, dim=-1)
    
    if positive_pairs:
        loss = 1.0 - cosine_sim
    else:
        loss = torch.relu(cosine_sim - margin)
    
    return torch.mean(loss)


def main(save_path, label_smooth=True):
    not_update = 0
    # exp_name should be defined or passed
    exp_name = save_path.split('/')[-1] if save_path else "default_exp"
    writer = SummaryWriter(f'runs/{exp_name}')
    
    print("Gathering image pairs...")
    image_pairs = []
    # ... (unchanged) ...
    for query_path in tqdm(glob(f'{DRONE_VIEW_FOLDER}/*/*/image-01.jpeg')):
        name = query_path.split('/')[-2]
        search_path = f"{IMAGE_FOLDER}/{name}.png"
        if os.path.exists(search_path): # Check if the corresponding search image exists
            image_pairs.append((query_path, search_path))
    print(f"Found {len(image_pairs)} valid pairs.")

    print("Loading models and processor...")
    # Load Vision Processors
    qprocessor = AutoImageProcessor.from_pretrained(VISION_MODEL_PATH)
    sprocessor = AutoImageProcessor.from_pretrained(VISION_MODEL_PATH, do_center_crop=False, do_resize=False)
    
    # *** NEW: Load Text Tokenizer (with trust_remote_code) ***
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True)
    
    # *** NEW: Instantiate new Encoder ***
    model = Encoder(
        vision_model_path=VISION_MODEL_PATH,
        text_model_name=TEXT_MODEL_NAME,
        proj_dim=PROJECTION_DIM
    ).to(DEVICE)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    img_to_text_dict = json.load(open('drone_text_single_long.json', 'r'))
    
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
            attention_mask = batch['attention_mask'].to(DEVICE) # Get attention mask
            local_indices = batch['index'].to(DEVICE)
            
            anchor_feats = model.query_forward(target_pixel_values)
            grid_feats = model.ref_forward(search_pixel_values)
            
            # Use the new feature_dim from the projector
            candidate_feats = grid_feats.reshape(-1, PROJECTION_DIM) 
            
            # *** NEW: Pass attention_mask to text_forward ***
            text_feats = model.text_forward(input_ids, attention_mask)
            
            # Label smoothing logic
            positive_indices = torch.zeros(local_indices.shape[0], local_indices.shape[0]*9, device=DEVICE)
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
            
            # *** NEW: Combine query and text features via fusion MLP ***
            combined_query_feats = model.forward_fusion(anchor_feats, text_feats)
            img_text_loss = info_nce_loss(combined_query_feats, candidate_feats, positive_indices)
            
            loss = img_text_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'img_text_loss': f'{img_text_loss.item():.4f}'})
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_dataloader) + i)
            
        avg_loss = total_loss / len(train_dataloader)
        
        model.eval()
        val_loss, txt_val_loss = validation(model, test_dataloader, epoch)
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Loss/val_img_epoch', val_loss, epoch)
        writer.add_scalar('Loss/val_txt_epoch', txt_val_loss, epoch)
        print(f"Epoch {epoch+1} finished. Average train Loss: {avg_loss:.4f}. Average test Loss: {txt_val_loss:.4f}")
        
        if txt_val_loss < min_loss:
            os.makedirs(save_path, exist_ok=True) # Ensure directory exists
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
    progress_bar = tqdm(loader, desc=f"Valid {epoch+1}/{NUM_EPOCHS}")
    total_loss = 0
    txt_total = 0
    for batch in progress_bar:
        target_pixel_values = batch['target_pixel_values'].to(DEVICE)
        search_pixel_values = batch['search_pixel_values'].to(DEVICE)
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE) # Get attention mask
        local_indices = batch['index'].to(DEVICE)
        
        batch_offsets = torch.arange(local_indices.shape[0], device=DEVICE) * 9
        positive_indices = local_indices + batch_offsets
        
        with torch.no_grad():
            anchor_feats = model.query_forward(target_pixel_values)
            grid_feats = model.ref_forward(search_pixel_values)
            # *** NEW: Pass attention_mask ***
            text_feats = model.text_forward(input_ids, attention_mask)
            
            candidate_feats = grid_feats.reshape(-1, PROJECTION_DIM)
            
            # Note: Your validation uses hard labels for info_nce
            loss = info_nce_loss(anchor_feats, candidate_feats, positive_indices)
            
            # *** NEW: Combine query and text features via fusion MLP ***
            combined_query_feats = model.forward_fusion(anchor_feats, text_feats)
            img_txt_loss = info_nce_loss(combined_query_feats, candidate_feats, positive_indices)
            
            total_loss += loss.item()
            txt_total += img_txt_loss.item() # Use .item()
            
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

        qprocessor = AutoImageProcessor.from_pretrained(VISION_MODEL_PATH)
        sprocessor = AutoImageProcessor.from_pretrained(VISION_MODEL_PATH, do_center_crop=False, do_resize=False)
        # *** NEW: Load correct tokenizer (with trust_remote_code) ***
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True)
        
        # *** NEW: Instantiate new Encoder ***
        encoder = Encoder(
            vision_model_path=VISION_MODEL_PATH,
            text_model_name=TEXT_MODEL_NAME,
            proj_dim=PROJECTION_DIM
        ).to(DEVICE)
        
        # Make sure this path is correct for the new model
        # This path might need to be updated to your new checkpoint directory
        model_path = "../ckpt/train_projected_bert/best.pth" # ASSUMED PATH
        if not os.path.exists(model_path):
             model_path = "../ckpt/train_add/best.pth" # Fallback to old path
        print(f"Loading model from {model_path}")
        encoder.load_state_dict(torch.load(model_path, map_location='cpu'))
        encoder.eval()

        res_search = {}
        res_fused_query = {} # *** NEW: Store fused query
        
        print("Setting up dataset and dataloader for eval...")
        
        eval_list = glob(f'{DRONE_VIEW_FOLDER}/*/*/image-01.jpeg')
        if len(eval_list) > 400:
             eval_list = eval_list[-400:]

        for img_path in tqdm(eval_list):
            name = img_path.split('/')[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            if os.path.exists(search_path):
                try:
                    query_image = Image.open(img_path).convert("RGB")
                    search_image = Image.open(search_path).convert("RGB")
                    search_image = search_image.crop((840,0, 3000, 2160)).resize((640, 640))
                    
                    text_name = name + '_01'
                    text_description = img_to_text_dict.get(text_name, "")
                    for noun in ['**', '\n', 'noun', 'phrases', 'Phrase', 'Noun', 'Summary', 'Environment', '32 tokens']:
                        text_description = text_description.replace(noun, '')

                    query_inputs = qprocessor(images=query_image, return_tensors="pt")['pixel_values'].to(DEVICE)
                    search_inputs = sprocessor(images=search_image, return_tensors="pt")['pixel_values'].to(DEVICE)
                    text_inputs = tokenizer(
                        text_description,
                        padding="max_length",
                        truncation=True,
                        max_length=128,
                        return_tensors="pt"
                    )
                    input_ids = text_inputs['input_ids'].to(DEVICE)
                    attention_mask = text_inputs['attention_mask'].to(DEVICE)
                    
                    with torch.no_grad():
                        anchor_feats = encoder.query_forward(query_inputs)
                        grid_feats = encoder.ref_forward(search_inputs)
                        # *** NEW: Pass attention_mask ***
                        text_feats = encoder.text_forward(input_ids, attention_mask)
                        
                        # *** NEW: Get fused feature ***
                        fused_query_feats = encoder.forward_fusion(anchor_feats, text_feats)
                        
                    res_search[name] = grid_feats.cpu().numpy()
                    res_fused_query[name] = fused_query_feats.cpu().numpy() # Save fused
                except Exception as e:
                    print(f"Error processing {name}: {e}")

        np.savez("eval_search.npz", **res_search)
        np.savez("eval_fused_query.npz", **res_fused_query) # *** NEW: Save fused
        print("Evaluation feature extraction complete.")
    else:
        # *** NEW: Load fused query features ***
        search_res = np.load("eval_search.npz")
        fused_query_res = np.load("eval_fused_query.npz")
        
        distances = json.load(open('distances.json','r'))
        test_num = 100
        test_list = [k for k in search_res.keys()]
        res = {}
        top1 = 0
        top5 = 0
        top10 = 0
        image_name_list = [k for k in fused_query_res.keys()]
        
        if not test_list:
            print("No evaluation data found in npz files.")
            return

        for key in tqdm(test_list):
            # *** NEW: Load fused feature ***
            fused_query_feature = fused_query_res[key]
            
            # Create candidate list
            ex_img_list = random.sample(test_list, min(test_num-1, len(test_list)-1))
            if key in ex_img_list:
                ex_img_list.remove(key)
            ex_img_list.append(key) # Add true positive at the end
            
            # Find true positives (within 1000m)
            candidate_indices = [len(ex_img_list) - 1] # The key itself is always a candidate
            for i, name in enumerate(ex_img_list[:-1]):
                if f'{name}.kml' not in distances or f'{key}.kml' not in distances[f'{name}.kml']:
                    continue
                if distances[f'{name}.kml'][f'{key}.kml'] < 1000:
                    candidate_indices.append(i)
            
            res[key] = []
            for img_name in ex_img_list:
                # *** NEW: Compare fused query to search grid ***
                cos_sim_grid = calcu_cos(fused_query_feature, search_res[img_name])
                res[key].append(cos_sim_grid)
            
            # Get mean similarity across 9 grid cells
            img_res = np.array(res[key]).mean(1).argsort()[-15:][::-1]
            
            # Check for top-k hits
            if any(cand in candidate_indices for cand in img_res[:1]):
                top1 += 1
            if any(cand in candidate_indices for cand in img_res[:5]):
                top5 += 1
            if any(cand in candidate_indices for cand in img_res[:10]):
                top10 += 1
                
        print(f"Top 1: {top1} / {len(test_list)} ({top1 / len(test_list) * 100:.2f}%)")
        print(f"Top 5: {top5} / {len(test_list)} ({top5 / len(test_list) * 100:.2f}%)")
        print(f"Top 10: {top10} / {len(test_list)} ({top10 / len(test_list) * 100:.2f}%)")

if __name__ == '__main__':
    # Define and create the save path
    exp_name = 'train_projected_nv_embed' # *** NEW: Updated exp_name
    save_dir = f'../ckpt/{exp_name}'
    
    # Check if dir exists, ask user if they want to overwrite
    if os.path.exists(save_dir):
        print(f"Experiment directory '{save_dir}' already exists.")
        # Re-using the directory for a new run
        print("Re-using directory. Logs and checkpoints might be overwritten.")
        # Or, you could stop:
        # print("Please choose a new 'exp_name'. Exiting.")
        # sys.exit(1)
    else:
        os.makedirs(save_dir)
        print(f"Created experiment directory: {save_dir}")

    # Run training
    # main(save_dir)
    
    # Run evaluation
    eval(True)  # Extract features
    eval(False) # Calculate metrics