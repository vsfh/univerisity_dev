import json
import os
os.environ['HF_HOME'] = '/home/SATA4T/gregory/hf_cache'
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor
import random
from glob import glob
from tqdm import tqdm

# Paths (replace with your actual paths)
image_folder = "/home/SATA4T/gregory/data/image_1024"  # Update this
drone_view_folder = "/home/SATA4T/gregory/data/drone_view"  # Update this
bbox_json_path = "/home/SATA4T/gregory/zero/retrieval/data/boxes_output.json"  # Update this
custom_cache_path = "/home/SATA4T/gregory/hf_cache"

# Load DINOv2 model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", cache_dir=custom_cache_path)
model = AutoModel.from_pretrained("facebook/dinov2-base", cache_dir=custom_cache_path).to(device)
model.eval()

# sig_str = 'google/siglip-base-patch16-224'
# model = SiglipModel.from_pretrained(sig_str, cache_dir=custom_cache_path).to(device)
# processor = SiglipProcessor.from_pretrained(sig_str, cache_dir=custom_cache_path)
# model.eval()

def extract_bbox_feature(siglip=False):
    def dilate_bbox(bbox, scale=1.2, img_width=None, img_height=None):
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        
        # Calculate new dimensions
        new_width = width * scale
        new_height = height * scale
        
        # Calculate center
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # New coordinates
        new_x_min = max(0, center_x - new_width / 2)
        new_y_min = max(0, center_y - new_height / 2)
        new_x_max = min(img_width, center_x + new_width / 2) if img_width else center_x + new_width / 2
        new_y_max = min(img_height, center_y + new_height / 2) if img_height else center_y + new_height / 2
        
        return [int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)]

    # Prepare to store features
    features_dict = {}
    bbox_crop = []
    num_patch_sqrt = 4
    h = 2160 // num_patch_sqrt
    for i in range(num_patch_sqrt):
        for j in range(num_patch_sqrt):
            bbox_crop.append([i*h, j*h, (i+1)*h, (j+1)*h])

    # Process each image and its bounding boxes
    for image_path in tqdm(glob('/home/SATA4T/gregory/data/image_1024/*.png')):
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = image.crop((840,0, 3000, 2160))

        img_name = image_path.split('/')[-1].split('.')[0]
        fea_list = []
        img_width, img_height = image.size
        
        # Process each bounding box
        for idx, bbox in enumerate(bbox_crop):
            # Dilate bbox
            dilated_bbox = dilate_bbox(bbox, scale=1.2, img_width=img_width, img_height=img_height)
            x_min, y_min, x_max, y_max = dilated_bbox
            
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            
            # Preprocess cropped image
            inputs = processor(images=cropped_image, return_tensors="pt").to(device)
            
            # Extract features
            with torch.no_grad():
                if not siglip:
                    outputs = model(**inputs)
                    features = outputs.last_hidden_state[:, 0, :]  # CLS token (1D feature)
                    features = features.cpu().numpy().flatten()  # Convert to 1D numpy array
                else:
                    features = model.get_image_features(**inputs)
                    features = features.cpu().numpy().flatten().astype(np.float32)
            fea_list.append(features)

            
        features_dict[img_name] = np.array(fea_list)
        
    if not siglip:  
        output_npz_path = "/home/SATA4T/gregory/data/dino_features_dict.npz"
    else:
        output_npz_path = "/home/SATA4T/gregory/data/siglip_features_dict.npz"

    # Save features to NPZ file
    np.savez(output_npz_path, **features_dict)
    print(f"Features saved to {output_npz_path}")
    
def extract_patch_feature(siglip=False):
    from transformers import Dinov2Config, Dinov2Model, AutoImageProcessor

    image_height, image_width = 224, 224

    checkpoint = "facebook/dinov2-base"

    # Create a new model with randomly initialized weights
    model_config = Dinov2Config.from_pretrained(checkpoint, image_size=(image_height, image_width), local_files_only=True)
    model = Dinov2Model(model_config).to(device)
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", size=image_height, do_resize=True, do_center_crop=False, local_files_only=True)
    features_dict = {}
    if not siglip:
        output_npz_path = '/home/SATA4T/gregory/data/dino_patch_feature.npz'
    else:
        output_npz_path = '/home/SATA4T/gregory/data/dino_drone_feature_siglip.npz'
    for image_path in tqdm(glob('/home/SATA4T/gregory/data/image_1024/*.png')[:110]):
        image = Image.open(image_path).convert("RGB")
        image = image.crop((840,0, 3000, 2160))
        # Preprocess cropped image
        inputs = image_processor(images=image, return_tensors="pt").to(device)

        # Extract features
        with torch.no_grad():
            if not siglip:
                outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 1:, :]  # CLS token (1D feature)
                features = features.cpu().numpy()  # Convert to 1D numpy array
            else:
                features = model.get_image_features(**inputs)
                features = features.cpu().numpy().flatten().astype(np.float32)
        # Store features with unique key (e.g., image_name_bbox_idx)
        key = image_path.split('/')[-1][:4]
        features_dict[key] = features
        
    np.savez(output_npz_path, **features_dict)
    print(f"Features saved to {output_npz_path}")

def extract_drone_feature(siglip=False):
    features_dict = {}
    if not siglip:
        output_npz_path = '/home/SATA4T/gregory/data/dino_drone_feature.npz'
    else:
        output_npz_path = '/home/SATA4T/gregory/data/siglip_drone_feature.npz'
    for image_path in tqdm(glob(f'{drone_view_folder}/*/*/image-01.jpeg')):
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess cropped image
        inputs = processor(images=image, return_tensors="pt").to(device)


        # Extract features
        with torch.no_grad():
            if not siglip:
                outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]  # CLS token (1D feature)
                features = features.cpu().numpy().flatten()  # Convert to 1D numpy array
            else:
                features = model.get_image_features(**inputs)
                features = features.cpu().numpy().flatten().astype(np.float32)
        # Store features with unique key (e.g., image_name_bbox_idx)
        key = image_path.split('/')[-2]
        features_dict[key] = features
        
    np.savez(output_npz_path, **features_dict)
    print(f"Features saved to {output_npz_path}")
   
def feature_align(test_num = 100):
    def calcu_cos(feature_query, feature_value):
        return np.dot(feature_query, feature_value) / (np.linalg.norm(feature_query) * np.linalg.norm(feature_value))
    feature1 = np.load('/home/SATA4T/gregory/data/dino_drone_feature.npz')
    feature2 = np.load('/home/SATA4T/gregory/data/dino_features_dict.npz')
    search_img_num = 200
    test_num  = 100
    test_list = [k for k in feature2.keys()][:100]
    res = {}
    image_name_list = [k for k in feature2.keys()]
    for key in tqdm(test_list):
        drone_feature = feature1[key]
        ex_img_list = random.sample(image_name_list, test_num-1)
        if key in ex_img_list:
            ex_img_list.remove(key)
            ex_img_list.append(image_name_list[-1])
        ex_img_list.append(key)
        res[key] = []
        for img_name in ex_img_list:
            res[key].append([calcu_cos(drone_feature, feature2[img_name][0,i]) for i in range(feature2[img_name].shape[1])])
        # break
    np.savez(f'/home/SATA4T/gregory/data/align_res_patch.npz', res)
    print('save ok')

def align_check(test_num = 100):
    res = np.load('/home/SATA4T/gregory/data/align_res_patch.npz', allow_pickle=True)
    res = res['arr_0'].item()
    top1 = 0
    top3 = 0
    top5 = 0
    for k in res.keys():
        rank_five = np.array(res[k]).max(1).argsort()[-5:][::-1]
        if test_num-1 in rank_five[:1]:
            top1 += 1
        if test_num-1 in rank_five[:3]:
            top3 += 1
        if test_num-1 in rank_five[:5]:  
            top5 += 1
    print(f'top1:{top1/len(res)}, top3:{top3/len(res)}, top5:{top5/len(res)}')
 
extract_bbox_feature(False)
extract_drone_feature(False)
# extract_patch_feature(False)
# feature_align(test_num = 100)

# align_check(test_num = 100)
