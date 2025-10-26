from PIL import Image
import numpy as np
import glob
import os
import json

import random
from typing import List, Dict

def generate_random_hex_color() -> str:
    """生成随机的十六进制颜色代码"""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def extend_rectangle_labels(n: int) -> List[Dict]:

    last_id = 0
    last_num = 0

    new_items = []
    for i in range(n+1):
        new_id = last_id + i
        new_num = last_num + i
        new_item = {
            "name": f"building{new_num}",
            "id": new_id,
            "color": generate_random_hex_color(),
            "type": "rectangle",
            "attributes": []
        }
        new_items.append(new_item)
    json.dump(new_items, open('a.json', 'w'))
    return new_items


def sam_predict():
    from lang_sam import LangSAM
    model = LangSAM()
    outpath = '/data/lab/yan/huzhang/huzhang/gregory/capture_img/res_sample'
    for img_path in glob.glob('/data/lab/yan/huzhang/huzhang/gregory/capture_img/image_512/*.jpg'):
        img_basename = img_path.split('/')[-1]
        image_pil = Image.open(img_path).convert("RGB")
        text_prompt = "building"
        results = model.predict([image_pil], [text_prompt])
        if len(results) > 0:
            out_json = os.path.join(outpath, img_basename.replace('.jpg','.json') )
            res = {}
            for key, item in results[0].items():
                if key in ['labels','masks']:
                    continue
                if isinstance(item, np.ndarray):
                    res[key] = item.tolist()
            json.dump(res,open(out_json, 'w'))
        # break

def floats_to_ints(nested_list):
    return [floats_to_ints(x) if isinstance(x, list) else int(x) for x in nested_list]

def show_res():
    import cv2
    json_dir = '../data/res_sample'
    img_dir = '../data/image_sample'
    for name in os.listdir(json_dir):
        json_path = f'{json_dir}/{name}'
        img_name = name.replace('.json', '.jpg')
        img_path = f'{img_dir}/{img_name}'
        boxes = json.load(open(json_path, 'r'))['boxes']
        boxes = floats_to_ints(boxes)

        img = cv2.imread(img_path)
        for box in boxes:
            cv2.rectangle(img, tuple(box[:2]), tuple(box[-2:]), color=(0,0,255),thickness=2)
        cv2.imshow('aaa', img)
        cv2.waitKey(0)

def trans_coco():
    import cv2
    json_dir = '../data/res_sample'
    coco_dir = '../data/coco_sample'
    for name in os.listdir(json_dir):

        json_path = f'{json_dir}/{name}'
        txt_name = name.replace('.json', '.txt')
        txt_path = f'{coco_dir}/{txt_name}'
        boxes = json.load(open(json_path, 'r'))['boxes']
        boxes = np.array(floats_to_ints(boxes))
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        a = []
        b = []
        index = -1
        building_index = int(name[:4])+1

        for i in range(len(x1)):
            cx = (x1[i]+x2[i])/2/1280
            cy = (y1[i]+y2[i])/2/720
            w = (x2[i]-x1[i])/1280
            h = (y2[i]-y1[i])/720
            if w > 0.8 and h > 0.8:
                continue
            if cx - w/2 < 0.5 and cx + w/2 > 0.5 and cy - h/2 < 0.5 and cy + h/2 >0.5 and index < 0:
                index = i
                a.append(f'{building_index} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n')
            else:
                a.append(f'0 {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n')

            b.append((cx*1280-640)**2+(cy*720-360)**2)
        if index == -1:
            index = len(a)
            cx = 1/2
            cy = 1/2
            w = 200/1280
            h = 200/720
            a.append(f'{building_index} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n')

        with open(txt_path, 'w') as f:
            for i in range(len(a)):
                f.writelines(f'{a[i]}')

            f.close()
trans_coco()