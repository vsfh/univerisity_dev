import torch
from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor
from PIL import Image
import numpy as np
import os
os.environ['HF_HOME'] = '/home/SATA4T/gregory/hf_cache'
from tqdm import tqdm
img_root = '/weka/data/lab/yan/huzhang/huzhang/gregory/code/X-VLM/data/image_1024'
device = "cuda" if torch.cuda.is_available() else "cpu"
custom_cache_path = "/home/SATA4T/gregory/hf_cache"
def clip_siglip_feature_extraction(image_folder = "image_512",output_folder = "features"):
    # Load models and processors
    clip_str = "openai/clip-vit-base-patch32"
    # "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_str, cache_dir=custom_cache_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_str, cache_dir=custom_cache_path)
    sig_str = 'google/siglip-base-patch16-224'
    # model_str = google/siglip-large-patch16-256
    siglip_model = SiglipModel.from_pretrained(sig_str, cache_dir=custom_cache_path).to(device)
    siglip_processor = SiglipProcessor.from_pretrained(sig_str, cache_dir=custom_cache_path)

    def extract_features(image_path, model, processor, model_name):
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            features = model.get_image_features(**inputs)
        return features.cpu().numpy().flatten().astype(np.float32)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for filename in tqdm(os.listdir(image_folder)):
        if any(filename.lower().endswith(ext) for ext in extensions):
            base_name = os.path.splitext(filename)[0]
            # if os.path.exists(os.path.join(output_folder, f"{base_name}_clip_l.npy")) and \
            #     os.path.exists(os.path.join(output_folder, f"{base_name}_siglip_l.npy")):
            #     continue
            image_path = os.path.join(image_folder, filename)
            
            # Extract features
            clip_features = extract_features(image_path, clip_model, clip_processor, "CLIP")
            siglip_features = extract_features(image_path, siglip_model, siglip_processor, "SIGLIP")
            
            # Save as NPY files
            np.save(os.path.join(output_folder, f"{base_name}_clip.npy"), clip_features)
            np.save(os.path.join(output_folder, f"{base_name}_siglip.npy"), siglip_features)

    # Clean up to free memory
    del clip_model, siglip_model
    torch.cuda.empty_cache()

def eva_clip_feature_extraction(image_folder = "image_512",output_folder = "features"):
    import open_clip

    # Load EVA-CLIP models
    eva_clip_l_model, _, eva_clip_l_preprocess = open_clip.create_model_and_transforms(
        "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k", cache_dir=custom_cache_path
    )
    eva_clip_l_model = eva_clip_l_model.to(device)

    eva_clip_b_model, _, eva_clip_b_preprocess = open_clip.create_model_and_transforms(
        'hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k', cache_dir=custom_cache_path
    )
    eva_clip_b_model = eva_clip_b_model.to(device)

    def extract_eva_clip_features(image_path, model, preprocess):
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
        return features.cpu().numpy().flatten().astype(np.float32)

    # Process images

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for filename in tqdm(os.listdir(image_folder)):

        if any(filename.lower().endswith(ext) for ext in extensions):
            base_name = os.path.splitext(filename)[0]
            image_path = os.path.join(image_folder, filename)
            if os.path.exists(os.path.join(output_folder, f"{base_name}_eva_clip_b.npy")) and \
                os.path.exists(os.path.join(output_folder, f"{base_name}_eva_clip_l.npy")):
                continue  
            print(base_name)          
            # Extract features with EVA-CLIP models
            eva_clip_l_features = extract_eva_clip_features(image_path, eva_clip_l_model, eva_clip_l_preprocess)
            eva_clip_b_features = extract_eva_clip_features(image_path, eva_clip_b_model, eva_clip_b_preprocess)
            
            # Save as NPY files
            np.save(os.path.join(output_folder, f"{base_name}_eva_clip_l.npy"), eva_clip_l_features)
            np.save(os.path.join(output_folder, f"{base_name}_eva_clip_b.npy"), eva_clip_b_features)

    # Clean up to free memory
    del eva_clip_l_model, eva_clip_b_model
    torch.cuda.empty_cache()

def clip_siglip_large_feature_extraction(image_folder = "image_512",output_folder = "features", large=False):
    # Load models and processors
    if large:
        clip_model_name = "openai/clip-vit-large-patch14"
        siglip_model_name = "google/siglip-large-patch16-256"
    else:
        clip_model_name = "openai/clip-vit-base-patch32"
        siglip_model_name = "google/siglip-base-patch16-224"        
    # "openai/clip-vit-base-patch32"
    # "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name, cache_dir=custom_cache_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name, cache_dir=custom_cache_path)
    # google/siglip-base-patch16-224
    # google/siglip-large-patch16-256
    siglip_model = SiglipModel.from_pretrained(siglip_model_name, cache_dir=custom_cache_path).to(device)
    siglip_processor = SiglipProcessor.from_pretrained(siglip_model_name, cache_dir=custom_cache_path)

    def extract_features_dict(image_path, model, processor, boxes):
        image = Image.open(image_path).convert("RGB")
        feature_dict = {}
        for i, box in enumerate(boxes):
            img = image.crop(box=(int(box[0]-0.2*(box[2]-box[0])), int(box[1]-0.2*(box[3]-box[1])), int(box[2]+0.2*(box[2]-box[0])), int(box[3]+0.2*(box[3]-box[1]))))
            with torch.no_grad():
                inputs = processor(images=img, return_tensors="pt").to(device)
                features = model.get_image_features(**inputs)
                feature_dict[f'{i}'] = features.cpu().numpy().flatten().astype(np.float32)
        return feature_dict

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    data = np.load(f'/data/lab/yan/huzhang/huzhang/gregory/data/test_data/final_out.npz', allow_pickle=True)
    res_dict = data['arr_0'].item()
    os.makedirs(output_folder, exist_ok=True)
    for filename in tqdm(os.listdir(image_folder)):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_path = os.path.join(image_folder, filename)
            base_name = os.path.splitext(filename)[0]
            boxes = res_dict[base_name]['boxes']
            if len(boxes) > 12:
                boxes = boxes[:12]
            # if os.path.exists(os.path.join(output_folder, f"{base_name}_siglip_l.npz")):
            #     continue
            # Extract features
            clip_features = extract_features_dict(image_path, clip_model, clip_processor, boxes)
            siglip_features = extract_features_dict(image_path, siglip_model, siglip_processor, boxes)
            
            # Save as NPY files
            if large:
                np.savez(os.path.join(output_folder, f"{base_name}_clip_l.npz"), clip_features)
                np.savez(os.path.join(output_folder, f"{base_name}_siglip_l.npz"), siglip_features)
            else:
                np.savez(os.path.join(output_folder, f"{base_name}_clip.npz"), clip_features)
                np.savez(os.path.join(output_folder, f"{base_name}_siglip.npz"), siglip_features)            
        # break

    # Clean up to free memory
    del clip_model, siglip_model
    torch.cuda.empty_cache()
    
def eva_clip_large_feature_extraction(image_folder = "image_512",output_folder = "features"):
    import open_clip

    # Load EVA-CLIP models
    eva_clip_l_model, _, eva_clip_l_preprocess = open_clip.create_model_and_transforms(
        "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k", cache_dir=custom_cache_path
    )
    eva_clip_l_model = eva_clip_l_model.to(device)

    eva_clip_b_model, _, eva_clip_b_preprocess = open_clip.create_model_and_transforms(
        'hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k', cache_dir=custom_cache_path
    )
    eva_clip_b_model = eva_clip_b_model.to(device)

    def extract_eva_clip_features(image_path, model, preprocess):
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
        return features.cpu().numpy().flatten().astype(np.float32)

    def extract_features_dict(image_path, model, processor, boxes):
        image = Image.open(image_path).convert("RGB")
        feature_dict = {}
        for i, box in enumerate(boxes):
            img = image.crop(box=(int(box[0]-0.2*(box[2]-box[0])), int(box[1]-0.2*(box[3]-box[1])), int(box[2]+0.2*(box[2]-box[0])), int(box[3]+0.2*(box[3]-box[1]))))
            with torch.no_grad():
                inputs = processor(img).unsqueeze(0).to(device)
                features = model.encode_image(inputs)
                feature_dict[f'{i}'] = features.cpu().numpy().flatten().astype(np.float32)
        return feature_dict
    # Process images

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    data = np.load(f'/data/lab/yan/huzhang/huzhang/gregory/data/test_data/final_out.npz', allow_pickle=True)
    res_dict = data['arr_0'].item()
    for filename in tqdm(os.listdir(image_folder)):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_path = os.path.join(image_folder, filename)
            base_name = os.path.splitext(filename)[0]
            boxes = res_dict[base_name]['boxes']
            if len(boxes) > 12:
                boxes = boxes[:12]
        
            # Extract features with EVA-CLIP models
            eva_clip_l_features = extract_features_dict(image_path, eva_clip_l_model, eva_clip_l_preprocess, boxes)
            eva_clip_b_features = extract_features_dict(image_path, eva_clip_b_model, eva_clip_b_preprocess, boxes)
            
            # Save as NPY files
            np.savez(os.path.join(output_folder, f"{base_name}_eva_clip_l.npz"), eva_clip_l_features)
            np.savez(os.path.join(output_folder, f"{base_name}_eva_clip_b.npz"), eva_clip_b_features)
    # Clean up to free memory
    del eva_clip_l_model, eva_clip_b_model
    torch.cuda.empty_cache()

def search_img(search_img_num, name):
    import random
    from glob import glob
    from tqdm import tqdm
    def calcu_cos(feature_query, feature_value):
        return np.dot(feature_query, feature_value) / (np.linalg.norm(feature_query) * np.linalg.norm(feature_value))
    img_list = glob(f'./features_dict/*{name}.npz')
    res = {}

    for img_path in tqdm(img_list[:-1]):
        img_name = img_path.split('/')[-1].split('_')[0]+f'{name}.npy'
        if os.path.exists(f'./features_query/{img_name}') is False:
            continue
            print(img_name)
        query = np.load(f'./features_query/{img_name}', allow_pickle=True)
        sim_list = []
        name_list = []
        ex_img_list = random.sample(img_list, search_img_num-1)
        if img_path in ex_img_list:
            ex_img_list.remove(img_path)
            ex_img_list.append(img_list[-1])
        ex_img_list.append(img_path)

        for npz_path in ex_img_list:
            feature_dict = np.load(npz_path, allow_pickle=True)['arr_0'].item()
            sim_value = [calcu_cos(feature, query) for _, feature in feature_dict.items()]
            sim_list.append(sim_value)
            name_list.append(npz_path.split('/')[-1].split('_')[0])

        # rank = np.argsort(-np.array(sim_list))[-1]
        # print('rank: ', rank, img_name)
        res[img_name] = {'name': name_list, 'sim': sim_list}
    np.savez(f'search_res{name}.npz', res)
    print('save ok')
    
        
if __name__ == '__main__':
    clip_siglip_large_feature_extraction(img_root, '/data/lab/yan/huzhang/huzhang/gregory/data/features_dict', large=True)
    clip_siglip_large_feature_extraction(img_root, '/data/lab/yan/huzhang/huzhang/gregory/data/features_dict', large=False)
    # clip_siglip_feature_extraction('first_site', 'features_query')
    # eva_clip_feature_extraction('first_site', 'features_query')
    eva_clip_large_feature_extraction(img_root, '/data/lab/yan/huzhang/huzhang/gregory/data/features_dict')
    # search_img(100, '_clip_l')
    # search_img(100, '_clip')
    # search_img(100, '_siglip')
    # search_img(100, '_siglip_l')
    # search_img(100, '_eva_clip_l')
    # search_img(100, '_eva_clip_b')