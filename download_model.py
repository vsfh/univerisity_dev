import torch
import os
os.environ['HF_TOKEN'] = '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'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/home/SATA4T/gregory/hf_cache'
device = "cuda" if torch.cuda.is_available() else "cpu"
custom_cache_path = "/home/SATA4T/gregory/hf_cache"
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
if __name__=='__main__':
    nvemb()