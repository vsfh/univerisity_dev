"""
測試腳本：在 DenseUAV 數據集上進行圖像到圖像檢索測試
只使用圖像特徵進行檢索，不使用文本描述

使用示例：
1. 使用 SigLIP 模型：
   python test_denseuav_retrieval.py \
     --model_path google/siglip-base-patch16-224 \
     --checkpoint_path path/to/checkpoint.pth \
     --query_folder /path/to/drone_view \
     --search_folder /path/to/satellite_images \
     --use_siglip \
     --test_num 100

2. 使用 DINOv2 模型：
   python test_denseuav_retrieval.py \
     --model_path facebook/dinov2-base \
     --query_folder /path/to/drone_view \
     --search_folder /path/to/satellite_images \
     --test_num 100

3. 從保存的特徵文件加載（跳過特徵提取）：
   python test_denseuav_retrieval.py \
     --model_path google/siglip-base-patch16-224 \
     --query_folder /path/to/drone_view \
     --search_folder /path/to/satellite_images \
     --load_features denseuav_features.npz \
     --use_siglip
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
import os
from glob import glob
import random
import json
import argparse
from pathlib import Path

# 嘗試導入 rasterio 用於處理多波段 TIF 圖像（可選）
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class ImageRetrievalEncoder(nn.Module):
    """圖像檢索編碼器，用於提取查詢和數據庫圖像的特徵"""
    def __init__(self, model_path, use_siglip=False):
        super().__init__()
        if use_siglip:
            # 使用 SigLIP 模型
            self.backbone = AutoModel.from_pretrained(model_path).vision_model
            self.pool = nn.AdaptiveAvgPool2d((3, 3))
            self.feature_dim = self.backbone.config.hidden_size
        else:
            # 使用其他視覺模型（如 DINOv2）
            self.backbone = AutoModel.from_pretrained(model_path)
            self.feature_dim = self.backbone.config.hidden_size if hasattr(self.backbone.config, 'hidden_size') else 768
        
        self.use_siglip = use_siglip
    
    def extract_query_feature(self, pixel_values):
        """提取查詢圖像特徵（單一向量）"""
        if self.use_siglip:
            pooled_features = self.backbone(pixel_values).pooler_output
            return pooled_features
        else:
            # DINOv2 或其他模型
            outputs = self.backbone(pixel_values)
            if hasattr(outputs, 'last_hidden_state'):
                # 使用 CLS token
                features = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            else:
                features = outputs[0][:, 0, :]
            return features
    
    def extract_search_feature(self, pixel_values):
        """提取搜索圖像特徵（可以是網格特徵或單一向量）"""
        if self.use_siglip:
            # 提取 3x3 網格特徵
            outputs = self.backbone(pixel_values, interpolate_pos_encoding=True)
            patch_tokens = outputs.last_hidden_state
            B, N, D = patch_tokens.shape
            H = W = int(N**0.5)
            patch_tokens_for_pooling = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
            pooled_features = self.pool(patch_tokens_for_pooling)
            final_features = pooled_features.flatten(2).permute(0, 2, 1)  # [B, 9, D]
            return final_features
        else:
            # 對於非 SigLIP 模型，也返回單一向量
            return self.extract_query_feature(pixel_values)


def load_image_safe(img_path):
    """
    安全地加載圖像，支持 TIF 格式（包括多波段和 16 位圖像）
    
    Args:
        img_path: 圖像文件路徑
        
    Returns:
        PIL Image 對象（RGB 模式）
    """
    try:
        # 首先嘗試使用 PIL 直接打開（適用於大多數情況）
        img = Image.open(img_path)
        
        # 檢查是否為多波段圖像（如衛星圖像）
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            # 多幀圖像，選擇第一幀
            img.seek(0)
        
        # 處理 16 位或更高位深度圖像
        if img.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
            # 轉換為 8 位 RGB
            img_array = np.array(img)
            if len(img_array.shape) == 2:
                # 灰度圖像，轉換為 RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[2] > 3:
                # 多波段圖像，選擇前三個波段作為 RGB
                img_array = img_array[:, :, :3]
            
            # 歸一化到 0-255 範圍
            if img_array.dtype != np.uint8:
                img_min, img_max = img_array.min(), img_array.max()
                if img_max > img_min:
                    img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_array = np.zeros_like(img_array, dtype=np.uint8)
            
            img = Image.fromarray(img_array)
        
        # 轉換為 RGB 模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
        
    except Exception as e:
        # 如果 PIL 無法處理，嘗試使用 rasterio（如果可用）
        if HAS_RASTERIO and (img_path.lower().endswith('.tif') or img_path.lower().endswith('.tiff')):
            try:
                with rasterio.open(img_path) as src:
                    # 讀取 RGB 波段（通常是前三個波段）
                    bands = min(3, src.count)
                    img_array = src.read(list(range(1, bands + 1)))
                    
                    # 轉置為 (H, W, C) 格式
                    if len(img_array.shape) == 3:
                        img_array = np.transpose(img_array, (1, 2, 0))
                    
                    # 如果是單波段，複製為 RGB
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif img_array.shape[2] == 1:
                        img_array = np.repeat(img_array, 3, axis=2)
                    
                    # 歸一化到 0-255
                    if img_array.dtype != np.uint8:
                        img_min, img_max = img_array.min(), img_array.max()
                        if img_max > img_min:
                            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            img_array = np.zeros_like(img_array, dtype=np.uint8)
                    
                    img = Image.fromarray(img_array)
                    return img
            except Exception as rasterio_error:
                raise Exception(f"無法使用 PIL 或 rasterio 加載圖像 {img_path}: PIL錯誤={e}, rasterio錯誤={rasterio_error}")
        else:
            raise Exception(f"無法加載圖像 {img_path}: {e}")


def calculate_cosine_similarity(query_feat, search_feat):
    """計算餘弦相似度"""
    if len(search_feat.shape) == 2:
        # 單一向量
        query_norm = query_feat / np.linalg.norm(query_feat)
        search_norm = search_feat / np.linalg.norm(search_feat)
        return np.dot(query_norm, search_norm)
    else:
        # 網格特徵 [9, D]
        query_norm = query_feat / np.linalg.norm(query_feat)
        search_norm = search_feat / np.linalg.norm(search_feat, axis=1, keepdims=True)
        similarities = np.dot(search_norm, query_norm[..., None]).flatten()
        return similarities.mean()  # 返回平均相似度


def extract_features(model, processor, image_paths, device, batch_size=32, is_query=True):
    """批量提取圖像特徵"""
    features_dict = {}
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="提取特徵"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_paths = []
            
            for img_path in batch_paths:
                try:
                    img = load_image_safe(img_path)
                    batch_images.append(img)
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"無法加載圖像 {img_path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # 處理圖像
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            
            # 提取特徵
            if is_query:
                batch_features = model.extract_query_feature(inputs['pixel_values'])
            else:
                batch_features = model.extract_search_feature(inputs['pixel_values'])
            
            # 轉換為 numpy 並保存
            batch_features = batch_features.cpu().numpy()
            
            for j, img_path in enumerate(valid_paths):
                key = os.path.basename(img_path).split('.')[0]
                # 處理不同維度的特徵
                if len(batch_features.shape) == 2:
                    # 2D: [batch_size, feature_dim]
                    features_dict[key] = batch_features[j]
                elif len(batch_features.shape) == 3:
                    # 3D: [batch_size, num_patches, feature_dim] (例如 3x3 網格)
                    features_dict[key] = batch_features[j]
                else:
                    # 其他情況
                    features_dict[key] = batch_features[j]
    
    return features_dict


def evaluate_retrieval(query_features, search_features, query_list, search_list, 
                       test_num=100, distances=None, distance_threshold=1000):
    """評估檢索性能"""
    def calcu_cos(query_feat, search_feat):
        """計算餘弦相似度"""
        if len(search_feat.shape) == 1:
            # 單一向量
            query_norm = query_feat / np.linalg.norm(query_feat)
            search_norm = search_feat / np.linalg.norm(search_feat)
            return np.dot(query_norm, search_norm)
        else:
            # 網格特徵
            query_norm = query_feat / np.linalg.norm(query_feat)
            search_norm = search_feat / np.linalg.norm(search_feat, axis=1, keepdims=True)
            similarities = np.dot(search_norm, query_norm[..., None]).flatten()
            return similarities.mean()
    
    top1 = 0
    top5 = 0
    top10 = 0
    top20 = 0
    
    # 如果提供了距離信息，用於計算候選項
    use_distance = distances is not None
    
    for query_key in tqdm(query_list, desc="評估檢索"):
        if query_key not in query_features:
            continue
        
        query_feat = query_features[query_key]
        
        # 隨機選擇候選圖像
        candidate_list = random.sample(search_list, min(test_num - 1, len(search_list) - 1))
        if query_key in candidate_list:
            candidate_list.remove(query_key)
        
        # 添加正確答案
        candidate_list.append(query_key)
        true_positive_idx = len(candidate_list) - 1
        
        # 如果使用距離信息，找出所有候選項
        candidate_indices = [true_positive_idx]
        if use_distance:
            for i, name in enumerate(candidate_list[:-1]):
                query_kml = f'{query_key}.kml'
                candidate_kml = f'{name}.kml'
                # 檢查距離字典結構：distances[candidate_kml][query_kml] 或 distances[query_kml][candidate_kml]
                if candidate_kml in distances and query_kml in distances[candidate_kml]:
                    if distances[candidate_kml][query_kml] < distance_threshold:
                        candidate_indices.append(i)
                elif query_kml in distances and candidate_kml in distances[query_kml]:
                    if distances[query_kml][candidate_kml] < distance_threshold:
                        candidate_indices.append(i)
        
        # 計算相似度
        similarities = []
        for candidate_key in candidate_list:
            if candidate_key not in search_features:
                similarities.append(-1.0)  # 如果特徵不存在，設為最低相似度
                continue
            search_feat = search_features[candidate_key]
            sim = calcu_cos(query_feat, search_feat)
            similarities.append(sim)
        
        # 排序並獲取排名
        similarities = np.array(similarities)
        ranked_indices = np.argsort(-similarities)  # 降序排列
        
        # 檢查 top-k
        if any(cand in ranked_indices[:1] for cand in candidate_indices):
            top1 += 1
        if any(cand in ranked_indices[:5] for cand in candidate_indices):
            top5 += 1
        if any(cand in ranked_indices[:10] for cand in candidate_indices):
            top10 += 1
        if any(cand in ranked_indices[:20] for cand in candidate_indices):
            top20 += 1
    
    total = len(query_list)
    results = {
        'top1': (top1, top1 / total * 100 if total > 0 else 0),
        'top5': (top5, top5 / total * 100 if total > 0 else 0),
        'top10': (top10, top10 / total * 100 if total > 0 else 0),
        'top20': (top20, top20 / total * 100 if total > 0 else 0),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='在 DenseUAV 數據集上測試圖像檢索')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路徑（可以是 HuggingFace 模型名稱或本地路徑）')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='可選的模型檢查點路徑（.pth 文件）')
    parser.add_argument('--query_folder', type=str, required=True,
                        help='查詢圖像文件夾路徑（無人機視圖）')
    parser.add_argument('--search_folder', type=str, required=True,
                        help='搜索圖像文件夾路徑（衛星圖像）')
    parser.add_argument('--query_pattern', type=str, default='*5/H80.JPG',
                        help='查詢圖像的文件模式（glob 模式）')
    parser.add_argument('--search_pattern', type=str, default='*5/H100.tif',
                        help='搜索圖像的文件模式（glob 模式）')
    parser.add_argument('--distances_json', type=str, default=None,
                        help='可選的距離 JSON 文件路徑')
    parser.add_argument('--distance_threshold', type=float, default=1000.0,
                        help='距離閾值（米）')
    parser.add_argument('--test_num', type=int, default=100,
                        help='每個查詢的候選圖像數量')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--use_siglip', action='store_true',
                        help='是否使用 SigLIP 模型架構')
    parser.add_argument('--device', type=str, default=None,
                        help='設備（cuda/cpu），默認自動選擇')
    parser.add_argument('--save_features', default=True,
                        help='是否保存提取的特徵')
    parser.add_argument('--load_features', type=str, default=None,
                        help='從文件加載特徵（.npz 文件）')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子')
    
    args = parser.parse_args()
    
    # 設置隨機種子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 設置設備
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"使用設備: {device}")
    
    # 加載距離信息（如果提供）
    distances = None
    if args.distances_json and os.path.exists(args.distances_json):
        print(f"加載距離信息: {args.distances_json}")
        with open(args.distances_json, 'r') as f:
            distances = json.load(f)
    
    # 加載模型
    print(f"加載模型: {args.model_path}")
    processor = AutoImageProcessor.from_pretrained(args.model_path)
    model = ImageRetrievalEncoder(args.model_path, use_siglip=args.use_siglip).to(device)
    
    # 加載檢查點（如果提供）
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"加載檢查點: {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))
    
    model.eval()
    
    # 獲取圖像列表
    print("收集圖像文件...")
    query_paths = list(glob(os.path.join(args.query_folder, args.query_pattern)))
    search_paths = list(glob(os.path.join(args.search_folder, args.search_pattern)))
    
    print(f"找到 {len(query_paths)} 個查詢圖像")
    print(f"找到 {len(search_paths)} 個搜索圖像")
    
    if len(query_paths) == 0 or len(search_paths) == 0:
        print("錯誤：未找到圖像文件！")
        return
    
    # 提取或加載特徵
    if args.load_features:
        print(f"從文件加載特徵: {args.load_features}")
        features_data = np.load(args.load_features, allow_pickle=True)
        query_features = features_data['query_features'].item() if 'query_features' in features_data else {}
        search_features = features_data['search_features'].item() if 'search_features' in features_data else {}
    else:
        print("提取查詢圖像特徵...")
        query_features = extract_features(model, processor, query_paths, device, 
                                         args.batch_size, is_query=True)
        
        print("提取搜索圖像特徵...")
        search_features = extract_features(model, processor, search_paths, device, 
                                         args.batch_size, is_query=False)
        
        if args.save_features:
            save_path = 'denseuav_features.npz'
            print(f"保存特徵到: {save_path}")
            np.savez(save_path, 
                    query_features=query_features, 
                    search_features=search_features)
    
    # 準備測試列表
    query_list = list(query_features.keys())
    search_list = list(search_features.keys())
    
    # 確保查詢和搜索列表有交集
    common_keys = set(query_list) & set(search_list)
    if len(common_keys) == 0:
        print("警告：查詢和搜索圖像沒有共同的鍵！")
        print("將使用所有查詢圖像進行測試...")
        test_query_list = query_list
    else:
        test_query_list = list(common_keys)
    
    print(f"使用 {len(test_query_list)} 個查詢進行測試")
    
    # 評估
    print("開始評估...")
    results = evaluate_retrieval(
        query_features, 
        search_features, 
        test_query_list, 
        search_list,
        test_num=args.test_num,
        distances=distances,
        distance_threshold=args.distance_threshold
    )
    
    # 打印結果
    print("\n" + "="*50)
    print("檢索結果:")
    print("="*50)
    print(f"Top-1 準確率: {results['top1'][0]} / {len(test_query_list)} ({results['top1'][1]:.2f}%)")
    print(f"Top-5 準確率: {results['top5'][0]} / {len(test_query_list)} ({results['top5'][1]:.2f}%)")
    print(f"Top-10 準確率: {results['top10'][0]} / {len(test_query_list)} ({results['top10'][1]:.2f}%)")
    print(f"Top-20 準確率: {results['top20'][0]} / {len(test_query_list)} ({results['top20'][1]:.2f}%)")
    print("="*50)


if __name__ == '__main__':
    main()

