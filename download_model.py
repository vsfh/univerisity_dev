import torch
import os
os.environ['HF_HOME'] = '/data/feihong/hf_cache'
device = "cuda" if torch.cuda.is_available() else "cpu"
custom_cache_path = "/data/feihong/hf_cache"
def clip_siglip(image_folder = "image_512",output_folder = "features", large=False):
    import open_clip
    from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor
    # Load models and processors
    if large:
        clip_model_name = "openai/clip-vit-large-patch14"
        siglip_model_name = "google/siglip-large-patch16-256"
    else:
        clip_model_name = "openai/clip-vit-base-patch32"
        siglip_model_name = "google/siglip-base-patch16-224"        
    # "openai/clip-vit-base-patch32"
    # "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name, cache_dir=custom_cache_path)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name, cache_dir=custom_cache_path)
    # google/siglip-base-patch16-224
    # google/siglip-large-patch16-256
    siglip_model = SiglipModel.from_pretrained(siglip_model_name, cache_dir=custom_cache_path)
    siglip_processor = SiglipProcessor.from_pretrained(siglip_model_name, cache_dir=custom_cache_path)
    
    eva_clip_l_model, _, eva_clip_l_preprocess = open_clip.create_model_and_transforms(
        "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k", cache_dir=custom_cache_path
    )

    eva_clip_b_model, _, eva_clip_b_preprocess = open_clip.create_model_and_transforms(
        'hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k', cache_dir=custom_cache_path
    )
    
def dino():
    from transformers import AutoImageProcessor, AutoModel
    from urllib.request import urlopen
    from PIL import Image
    import timm
    import torch.nn.functional as F

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")


    img = Image.open(urlopen(
        'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    ))
    print(img.size)
    
    model = timm.create_model(
        'vit_small_plus_patch16_dinov3.lvd1689m',
        pretrained=True,
        features_only=True,
    )
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

    last_feature_map = output[-1]  # Shape: [1, 384, 16, 16]
    feature_vector_gap = F.adaptive_avg_pool2d(last_feature_map, (1, 1))  # Shape becomes [1, 384, 1, 1]
    feature_vector_gap = feature_vector_gap.view(feature_vector_gap.size(0), -1)

def qwen():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    # default: Load the model on the available device(s)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    
def nvemb():
    from transformers import AutoImageProcessor, AutoModel
    model_name = "nvidia/NV-Embed-v1"
    text_encoder = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True # CRITICAL for this model
    )

def download_vision_models(cache_dir=None, models_to_download=None):
    """
    下載多種視覺語言模型到本地緩存目錄
    
    Args:
        cache_dir (str): 模型緩存目錄，如果為 None 則使用 custom_cache_path
        models_to_download (list): 要下載的模型列表，可選值：
            - 'siglip': SigLIP 模型
            - 'clip': CLIP 模型
            - 'openclip': OpenCLIP 模型
            - 'evaclip': EVA-CLIP 模型
            - 'blip2': BLIP-2 模型
            如果為 None，則下載所有模型
    
    Returns:
        dict: 下載的模型和處理器字典
    """
    from transformers import (
        CLIPModel, CLIPProcessor, 
        SiglipModel, SiglipProcessor,
        Blip2Processor, Blip2ForConditionalGeneration,
        AutoModel, AutoImageProcessor
    )
    import open_clip
    from tqdm import tqdm
    
    if cache_dir is None:
        cache_dir = custom_cache_path
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # 定義所有可用的模型
    all_models = {
        'siglip': {
            'base': 'google/siglip-base-patch16-224',
            # 'large': 'google/siglip-large-patch16-256'
        },
        'clip': {
            'base': 'openai/clip-vit-base-patch32',
            # 'large': 'openai/clip-vit-large-patch14'
        },
        'openclip': {
            'base': 'ViT-B-32',
            # 'large': 'ViT-L-14'
        },
        'evaclip': {
            'base': 'hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k',
            # 'large': 'hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k'
        },
        'blip2': {
            'base': 'Salesforce/blip2-opt-2.7b',
            # 'large': 'Salesforce/blip2-opt-6.7b'
        }
    }
    
    # 如果未指定，則下載所有模型
    if models_to_download is None:
        models_to_download = list(all_models.keys())
    
    downloaded_models = {}
    
    print(f"開始下載模型到: {cache_dir}")
    print(f"將下載以下模型: {', '.join(models_to_download)}")
    
    # 下載 SigLIP
    if 'siglip' in models_to_download:
        print("\n[1/5] 下載 SigLIP 模型...")
        for size, model_name in all_models['siglip'].items():
            try:
                print(f"  下載 SigLIP {size}: {model_name}")
                model = SiglipModel.from_pretrained(model_name, cache_dir=cache_dir)
                processor = SiglipProcessor.from_pretrained(model_name, cache_dir=cache_dir)
                downloaded_models[f'siglip_{size}'] = {'model': model, 'processor': processor}
                print(f"  ✓ SigLIP {size} 下載完成")
            except Exception as e:
                print(f"  ✗ SigLIP {size} 下載失敗: {e}")
    
    # 下載 CLIP
    if 'clip' in models_to_download:
        print("\n[2/5] 下載 CLIP 模型...")
        for size, model_name in all_models['clip'].items():
            try:
                print(f"  下載 CLIP {size}: {model_name}")
                model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
                processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
                downloaded_models[f'clip_{size}'] = {'model': model, 'processor': processor}
                print(f"  ✓ CLIP {size} 下載完成")
            except Exception as e:
                print(f"  ✗ CLIP {size} 下載失敗: {e}")
    
    # 下載 OpenCLIP
    if 'openclip' in models_to_download:
        print("\n[3/5] 下載 OpenCLIP 模型...")
        for size, model_name in all_models['openclip'].items():
            try:
                print(f"  下載 OpenCLIP {size}: {model_name}")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, 
                    pretrained='openai',
                    cache_dir=cache_dir
                )
                tokenizer = open_clip.get_tokenizer(model_name)
                downloaded_models[f'openclip_{size}'] = {
                    'model': model, 
                    'preprocess': preprocess,
                    'tokenizer': tokenizer
                }
                print(f"  ✓ OpenCLIP {size} 下載完成")
            except Exception as e:
                print(f"  ✗ OpenCLIP {size} 下載失敗: {e}")
    
    # 下載 EVA-CLIP
    if 'evaclip' in models_to_download:
        print("\n[4/5] 下載 EVA-CLIP 模型...")
        for size, model_name in all_models['evaclip'].items():
            try:
                print(f"  下載 EVA-CLIP {size}: {model_name}")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name,
                    cache_dir=cache_dir
                )
                tokenizer = open_clip.get_tokenizer(model_name)
                downloaded_models[f'evaclip_{size}'] = {
                    'model': model,
                    'preprocess': preprocess,
                    'tokenizer': tokenizer
                }
                print(f"  ✓ EVA-CLIP {size} 下載完成")
            except Exception as e:
                print(f"  ✗ EVA-CLIP {size} 下載失敗: {e}")
    
    # 下載 BLIP-2
    if 'blip2' in models_to_download:
        print("\n[5/5] 下載 BLIP-2 模型...")
        for size, model_name in all_models['blip2'].items():
            try:
                print(f"  下載 BLIP-2 {size}: {model_name}")
                processor = Blip2Processor.from_pretrained(model_name, cache_dir=cache_dir)
                model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                downloaded_models[f'blip2_{size}'] = {'model': model, 'processor': processor}
                print(f"  ✓ BLIP-2 {size} 下載完成")
            except Exception as e:
                print(f"  ✗ BLIP-2 {size} 下載失敗: {e}")
    
    print(f"\n下載完成！共下載 {len(downloaded_models)} 個模型")
    return downloaded_models

if __name__=='__main__':
    download_vision_models()