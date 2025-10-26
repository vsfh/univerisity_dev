import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from tqdm import tqdm
import os
from glob import glob

# --- Configuration ---
model_path = "/home/SATA4T/gregory/hf_cache/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
NUM_EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
drone_view_folder = "/home/SATA4T/gregory/data/drone_view"
image_folder = "/home/SATA4T/gregory/data/image_1024"
    
# --- 1. Custom Dataset for Image Pairs ---
# This dataset loads a small target image and a larger search image.
# It applies the necessary transformations required by the SigLIP model.

class TargetSearchDataset(Dataset):
    """
    PyTorch Dataset for loading pairs of (small_target_image, large_search_image).
    """
    def __init__(self, image_pairs_df, processor):
        """
        Args:
            image_pairs_df (pd.DataFrame): DataFrame with 'target_path' and 'search_path' columns.
            processor (AutoImageProcessor): The image processor from Hugging Face for the model.
        """
        self.image_paths = image_pairs_df
        self.qprocessor = processor[0]
        self.sprocessor = processor[1]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load images from paths
        query_path = self.image_paths[idx][0]
        search_path = self.image_paths[idx][1]

        try:
            query_image = Image.open(query_path).convert("RGB")
            search_image = Image.open(search_path).convert("RGB")
            search_image = search_image.crop((840,0, 3000, 2160))
            search_image = search_image.resize((640,640))
            
        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping this item.")
            # Return a dummy item or handle this error as appropriate
            return self.__getitem__((idx + 1) % len(self))


        # Process images using the pre-trained model's processor.
        # This handles resizing, normalization, and tensor conversion.
        # We process them separately to handle potentially different resolutions
        # before the processor resizes them to the model's expected input size.
        query_inputs = self.qprocessor(images=query_image, return_tensors="pt")
        search_inputs = self.sprocessor(images=search_image, return_tensors="pt")

        return {
            'target_pixel_values': query_inputs['pixel_values'][0],
            'search_pixel_values': search_inputs['pixel_values'][0]
        }

class SearchEncoder(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path).vision_model
        feature_dim = (640//16)**2
        self.classifier = nn.Linear(feature_dim, num_labels)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values, interpolate_pos_encoding=True)
        cls_token_output = outputs.last_hidden_state
        
        origin_token = cls_token_output.permute(0, 2, 1)
        
        output = self.classifier(origin_token).permute(0, 2, 1)
        
        return output
# --- 2. Contrastive Loss Function (InfoNCE) ---

def info_nce_loss(anchor_feat, positive_feat, negative_feats, temperature=0.07):
    """
    Calculates the InfoNCE loss for contrastive learning.

    Args:
        anchor_feat (torch.Tensor): Feature of the small target image. Shape: [feature_dim]
        positive_feat (torch.Tensor): Feature of the central patch. Shape: [feature_dim]
        negative_feats (torch.Tensor): Features of all other patches. Shape: [num_negatives, feature_dim]
        temperature (float): Temperature hyperparameter for scaling logits.
    """
    # Combine positive and negative features for logit calculation
    # Reshape for broadcasting:
    # Anchor: [1, feature_dim]
    # Positive: [1, feature_dim]
    # Negatives: [num_negatives, feature_dim]
    all_candidate_feats = torch.cat([positive_feat.unsqueeze(0), negative_feats], dim=0)

    # Calculate cosine similarity between the anchor and all candidates
    # The result `logits` will have shape [1 + num_negatives]
    logits = F.cosine_similarity(anchor_feat.unsqueeze(0), all_candidate_feats, dim=1) / temperature

    # The target label is always the first one (the positive sample), so the index is 0.
    # We create a dummy target tensor for cross_entropy.
    labels = torch.zeros(1, dtype=torch.long, device=anchor_feat.device)

    # F.cross_entropy expects logits of shape [batch_size, num_classes]
    # Here, batch_size=1 and num_classes = 1 + num_negatives
    return F.cross_entropy(logits.unsqueeze(0), labels)


# --- 3. Main Training Script ---

def main():


    image_pairs = []
    for query_path in tqdm(glob(f'{drone_view_folder}/*/*/image-01.jpeg')):
        name = query_path.split('/')[-2]
        search_path = f"{image_folder}/{name}.png"
        image_pairs.append((query_path, search_path, ))


    # --- Model and Processor Setup ---
    print("Loading models and processor...")
    qprocessor = AutoImageProcessor.from_pretrained(model_path)
    sprocessor = AutoImageProcessor.from_pretrained(model_path, do_center_crop=False, do_resize=False)
    

    # 1. Query Tower (Frozen)
    query_encoder = AutoModel.from_pretrained(model_path).vision_model.to(DEVICE)
    query_encoder.eval() # Set to evaluation mode
    for param in query_encoder.parameters():
        param.requires_grad = False

    # 2. Search Tower (To be fine-tuned)
    search_encoder = SearchEncoder(model_path, 9).to(DEVICE)
    search_encoder.train() # Set to training mode

    # --- Data Loading ---
    print("Setting up dataset and dataloader...")
    train_dataset = TargetSearchDataset(image_pairs_df=image_pairs, processor=(qprocessor, sprocessor))
    # We use a batch size of 1 in the dataloader because the loss is calculated
    # per-image (one positive vs many negatives from a single search image).
    # You can implement a more complex collate_fn to batch this if needed.
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(search_encoder.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        
        # Use tqdm for a nice progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch in progress_bar:
            # Move data to the selected device
            target_pixel_values = batch['target_pixel_values'].to(DEVICE)
            search_pixel_values = batch['search_pixel_values'].to(DEVICE)

            # Since batch size is 1, we squeeze the batch dimension
            target_pixel_values = target_pixel_values.squeeze(0)
            search_pixel_values = search_pixel_values.squeeze(0)

            # --- Forward Pass ---
            with torch.no_grad(): # No gradients needed for the query tower
                anchor_feat = query_encoder(target_pixel_values.unsqueeze(0)).pooler_output.squeeze(0)

            # The search encoder returns all patch features + [CLS] token
            # output shape: [1, num_patches + 1, feature_dim]
            patch_feats = search_encoder(search_pixel_values.unsqueeze(0)).squeeze(0)
            
            # Total patches = 14 * 14 = 196.
            grid_size = int((patch_feats.shape[0])**0.5)
            
            # The patches are flattened in row-major order.
            # The center patch's index is at the center of this 1D array.
            center_patch_idx = (grid_size * (grid_size // 2)) + (grid_size // 2)

            positive_feat = patch_feats[center_patch_idx]
            
            # Create a mask to select all other patches as negatives
            negative_mask = torch.ones(patch_feats.shape[0], dtype=torch.bool)
            negative_mask[center_patch_idx] = False
            negative_feats = patch_feats[negative_mask]

            # --- Loss Calculation & Backward Pass ---
            loss = info_nce_loss(anchor_feat, positive_feat, negative_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    print("Training complete.")
    # --- Save the fine-tuned model ---
    # You would typically save the state_dict of the search_encoder
    # search_encoder.save_pretrained("my_finetuned_siglip_search_encoder")

if __name__ == '__main__':
    main()