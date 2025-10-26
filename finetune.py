import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
os.environ['HF_HOME'] = '/home/SATA4T/gregory/hf_cache'
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
# --- 1. Configuration ---
MODEL_NAME = "facebook/dinov2-base"  # You can choose 'dinov2-small', 'dinov2-base', 'dinov2-large', etc.
DATASET_ID = "cifar10"              # Example dataset (replace with your custom image dataset, e.g., 'imagefolder')
NUM_LABELS = 16                     # Number of classes in your dataset (Cifar10 has 10)
BATCH_SIZE = 8
LEARNING_RATE = 1e-6
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_cache_path = "/home/SATA4T/gregory/hf_cache"

class RetrievalHead(nn.Module):
    def __init__(self, input_dim=768, intermediate_size=150, output_size=16):
        super(RetrievalHead, self).__init__()
        self.input_dim = input_dim
        self.intermediate_size = intermediate_size
        self.output_size = output_size

        # Lightweight single-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=1,  # Single head for fewer parameters
            dropout=0.1,
            batch_first=True
        )
        # Linear layer to transform to final output size
        self.linear = nn.Linear(intermediate_size * input_dim, output_size * input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)  # Normalize output for stability

    def forward(self, x):
        # Input x: (batch_size, n, 768)
        batch_size, n, dim = x.shape

        # Step 1: Adaptive pooling to reduce n to intermediate_size
        assert n == self.intermediate_size, 'error shape'

        # Step 2: Apply single-head attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output  # Residual connection
        # Shape remains: (batch_size, intermediate_size, 768)

        # Step 3: Flatten and apply linear transformation
        x = x.reshape(batch_size, -1)  # (batch_size, intermediate_size * 768)
        x = self.linear(x)  # (batch_size, output_size * 768)
        x = x.reshape(batch_size, self.output_size, self.input_dim)  # (batch_size, 16, 768)

        # Step 4: Normalize output
        x = self.layer_norm(x)

        return x
    
# --- 2. Custom Model Class for Linear Probing ---
class SimpleDINOv2Dataset(Dataset):
    def __init__(self, image_dir, mode):
        self.image_dir = image_dir
        self.processor = AutoImageProcessor.from_pretrained(f"{custom_cache_path}/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415", 
                                                            do_center_crop=False, do_resize=False, cache_dir=custom_cache_path)
        image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        if mode == 'train':
            self.image_paths = image_paths[:-200]
        else:
            self.image_paths = image_paths[-200:]
        self.gt_tensor_dict = np.load("/home/SATA4T/gregory/data/dino_features_dict.npz")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = image.crop((840,0, 3000, 2160))
        image = image.resize((560,560))

        img_name = self.image_paths[idx].split('/')[-1].split('.')[0]
        gt_tensor = self.gt_tensor_dict[img_name]

        processed = self.processor(image, return_tensors="pt")
        
        return {
            'pixel_values': processed['pixel_values'].squeeze(0),
            'image_path': img_name,
            'labels': torch.tensor(gt_tensor, dtype=torch.float)
        }
    
class DINOv2LinearClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        # Load the DINOv2 model (Vision Transformer)
        # Use AutoModel for the backbone, not AutoModelForImageClassification
        self.backbone = AutoModel.from_pretrained(f"{custom_cache_path}/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415", cache_dir=custom_cache_path)
        
        # Determine the feature dimension (hidden size) of the model
        # ViT-Base has 768, ViT-Small has 384, ViT-Large has 1024, etc.
        feature_dim = (560//14)**2
        
        # Define the new linear classification head
        self.classifier = nn.Linear(feature_dim, num_labels)
        
        # --- Freeze the Backbone ---
        # print(f"Freezing all parameters in the DINOv2 backbone: {model_name}")
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
            
        print(f"The only trainable layer is the linear head with {self.classifier.out_features} output features.")

    def forward(self, pixel_values):
        # 1. Get features from the backbone
        # We pass pixel_values and get the model output
        # DINOv2's output includes the CLS token and all patch tokens.
        # By default, for classification, we typically use the CLS token's output.
        # In ViT-style models, this is the first token of the last_hidden_state.
        outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        
        # Extract the CLS token (the first token) from the last layer's hidden state
        # Shape: (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size)
        cls_token_output = outputs.last_hidden_state[:, 1:, :] 
        
        origin_token = cls_token_output.permute(0, 2, 1)
        
        output = self.classifier(origin_token).permute(0, 2, 1)
        
        return output

# --- 3. Data Loading and Preprocessing ---

def get_data_loaders(batch_size):
    # Load the dataset
    train_dataset = SimpleDINOv2Dataset('/home/SATA4T/gregory/data/image_1024', 'train')
    eval_dataset = SimpleDINOv2Dataset('/home/SATA4T/gregory/data/image_1024', 'test')


    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        shuffle=False, 
        batch_size=batch_size
    )
    
    return train_dataloader, eval_dataloader

# --- 4. Training and Evaluation Functions ---
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
    assert tensor1.shape[2] == 768, "Last dimension must be 768"
    
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

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        
        # Forward pass
        logits = model(pixel_values)
        
        # Compute loss
        loss = cosine_similarity_loss(logits, labels) + 0.1*criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if (step) % 50 == 0:
            print(f"Epoch {epoch+1}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"\n--- Epoch {epoch+1} finished. Average Training Loss: {avg_loss:.4f} ---")
    return avg_loss

def inference():
    _, eval_loader = get_data_loaders(1)

    model = DINOv2LinearClassifier(MODEL_NAME, NUM_LABELS).to(DEVICE)
    model.load_state_dict(torch.load("model_8.pth"))
    model.eval()
    res = {}
    
    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            img_name = batch["image_path"][0]
            logits = model(pixel_values)[0]
            res[img_name] = logits.cpu().numpy()
    np.savez('/home/SATA4T/gregory/data/dino_finetune_feature.npz', **res)
    return

def evaluate(res_path,save_path):
    def calcu_cos(feature_query, feature_value):
        return np.dot(feature_query, feature_value) / (np.linalg.norm(feature_query) * np.linalg.norm(feature_value))
    feature1 = np.load('/home/SATA4T/gregory/data/dino_drone_feature.npz')
    feature2 = np.load(res_path)
    search_img_num = 200
    image_dir = '/home/SATA4T/gregory/data/image_1024'
    test_num  = 100
    image_paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ][-100:]
    res = {}
    image_name_list = [img_path.split('/')[-1].split('.')[0] for img_path in image_paths]
    for img_path in tqdm(image_paths):
        key = img_path.split('/')[-1].split('.')[0]
        
        drone_feature = feature1[key]
        ex_img_list = random.sample(image_name_list, test_num-1)
        if key in ex_img_list:
            ex_img_list.remove(key)
            ex_img_list.append(image_name_list[-1])
        ex_img_list.append(key)
        res[key] = []
        for img_name in ex_img_list:
            res[key].append([calcu_cos(drone_feature, feature2[img_name][i]) for i in range(feature2[img_name].shape[0])])
        # break
    np.savez(save_path, res)
    print('save ok')
    return

# --- 5. Main Execution ---

def main():
    print(f"Using device: {DEVICE}")
    
    # Initialize the model, which includes freezing the backbone
    model = DINOv2LinearClassifier(MODEL_NAME, NUM_LABELS).to(DEVICE)
    
    # Get DataLoaders
    train_loader, eval_loader = get_data_loaders(BATCH_SIZE)
    
    # Only pass trainable parameters to the optimizer
    # This is a crucial step for linear probing!
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)
    
    # Loss function (Cross-Entropy for classification)
    criterion = nn.MSELoss()
    
    print("\nStarting Training...\n")
    
    best_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
        
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"model_{epoch}.pth")
        print(f"Model saved! New best accuracy: {best_accuracy:.4f}")

    print(f"\nTraining complete. Best validation accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    # main()
    inference()
    # evaluate('/home/SATA4T/gregory/data/dino_finetune_feature.npz', '1.npz')
    # evaluate('/home/SATA4T/gregory/data/dino_features_dict.npz', '2.npz')