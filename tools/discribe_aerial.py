import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/home/SATA4T/gregory/hf_cache'

from glob import glob
import json
from tqdm import tqdm

def qwen3():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    # default: Load the model on the available device(s)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen3-VL-2B-Instruct",
    #     dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    res = {}
    numbers = ['01','21','31','41','51']
    for img_path in tqdm(glob(f"/home/SATA4T/gregory/data/drone_view/*/*/image-01.jpeg")):
        index = img_path.split('/')[-2]
        message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "all images indicate the same building describe distinctive detail within 128 tokens focus on roof"},
                ],
            }
        
        for number in numbers:
            new_img_path = img_path.replace('01', number)
            if not os.path.exists(new_img_path):
                continue
            message['content'].append({"type": "image", "image": new_img_path })


        # Preparation for inference
        inputs = processor.apply_chat_template(
            [message],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        res[index] = output_text[0]
        
    json.dump(res, open('drone_text_long.json','w'))

def qwen3_single():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    # default: Load the model on the available device(s)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen3-VL-2B-Instruct",
    #     dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
    building_des = json.load(open('../ckpt/drone_text_long.json', 'r'))
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    # numbers = ['01','21','31','41','51']
    for img_path in tqdm(glob(f"/home/SATA4T/gregory/data/drone_view/*/*/image-01.jpeg")):
        index = img_path.split('/')[-2]
        if os.path.exists(f'../ckpt/text/{index}.json'):
            continue
        res = {}
        dir_path = os.path.dirname(img_path)
        numbers = [name.split('-')[-1][:2] for name in os.listdir(dir_path) if name.startswith('image-')]
        print(numbers)
        former_des = building_des[index]
        message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"summarize following building description with several noun phrase within 32 tokens, then describe environment except building within 32 tokens: <{former_des}>. Please output only the final answer. Do not output any tags that appear in my prompt, such as <summarize> or <32 tokens>."},
                ],
            }
        
        for number in numbers:
            new_img_path = img_path.replace('01', number)
            if not os.path.exists(new_img_path):
                continue
            message['content'].append({"type": "image", "image": new_img_path })
            name = number

            # Preparation for inference
            inputs = processor.apply_chat_template(
                [message],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=64)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_text)
            res[name] = output_text[0]
        
        json.dump(res, open(f'../ckpt/text/{index}.json','w'))

def qwen2():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch
    model_path = '/home/SATA4T/gregory/hf_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3/'
    # default: Load the model on the available device(s)
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_path, torch_dtype="auto", device_map="auto"
    # )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path)

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    res = {}
    numbers = ['01','21','31','41','51']
    for img_path in tqdm(glob(f"/home/SATA4T/gregory/data/drone_view/*/*/image-01.jpeg")):
        index = img_path.split('/')[-2]
        message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "all images indicate the same region describe distinctive region detail within 128 tokens focus on roof"},
                ],
            }
        
        for number in numbers:
            new_img_path = img_path.replace('01', number)
            if not os.path.exists(new_img_path):
                continue
            message['content'].append({"type": "image", "image": new_img_path })

        text = processor.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info([message])
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        res[index] = output_text[0]
        
    json.dump(res, open('drone_text.json','w'))

if __name__=='__main__':
    # qwen3()
    qwen3_single()