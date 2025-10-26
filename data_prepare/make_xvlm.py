import json
import os
import numpy as np
from glob import glob
from pathlib import Path

def make_data_list():
    img_width = 3840
    img_height = 2160
    new_items = []
    bbox_dict = {}
    for json_path in glob(r'D:\intern\labels\train\*.txt'):
        json_path = Path(json_path)
        json_name = json_path.name
        index = json_name.replace('.txt', '')
        if not os.path.exists(fr'D:\intern\data\drone_text_sec\{index}.json'):
            continue
        bbox_data = np.loadtxt(json_path)
        bbox_dict[index] = {}
        bbox_dict[index]['box'] = []
        if len(bbox_data.shape) == 1:
            bbox_data = np.array([bbox_data])
        box_certain = False
        for bbox in bbox_data:
            cls = int(bbox[0])
            bbox = tuple(bbox)
            x_center, y_center, w, h = bbox[1], bbox[2], bbox[3], bbox[4]
            x1 = int((x_center - w / 2) * img_width)
            y1 = int((y_center - h / 2) * img_height)
            x2 = int((x_center + w / 2) * img_width)
            y2 = int((y_center + h / 2) * img_height)
            if cls > 0:
                bbox_dict[index][cls] = [x1, y1, x2-x1, y2-y1]
                box_certain = True
            else:
                bbox_dict[index]['box'].append([x1, y1, x2-x1, y2-y1])
        if not box_certain:
            bbox_dict[index][cls] = [img_width/2, img_height/2, img_width/5, img_height/4]
            print(json_path)

        text_json = json.load(open(fr'D:\intern\data\drone_text_sec\{index}.json', 'r'))
        res_desc = ''
        for key, item in text_json.items():
            if isinstance(item, str):
                res_desc += item + ' '
            if isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, str):
                        res_desc += sub_item + ' '
        new_items.append({
            'image_path': json_name.replace('.txt', '.jpg'),
            'text': res_desc,
            'bboxes': index,
        })
    json.dump(new_items, open(r'data_list.json', 'w'))
    json.dump(bbox_dict, open(r'bbox_dict.json', 'w'))
    return new_items

def count():
    cnt = 0
    for txt_path in glob(r'D:\intern\labels\train\*.txt'):
        data = np.loadtxt(txt_path)
        cnt += data.shape[0]
    print(cnt)
    
def make_train_list():
    items = os.listdir(r'D:\intern\University-Release\train\drone')
    train_items = []
    for item in items:
        train_items.append(item)
    json.dump(train_items, open(r'train_id.json', 'w'))
    print(len(train_items))

# count()
make_train_list()