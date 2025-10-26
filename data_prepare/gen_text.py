import base64
import glob
from openai import OpenAI
import base64
import random
import json

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 将xxxx/eagle.png替换为你本地图像的绝对路径
def call_with_messages(index):
    user_message = {
        "role": "user",
        "content": [],
    }
    
    # img_list = glob.glob(fr"D:\intern\University-Release\test\gallery_drone\{index}\*.jpeg")
    img_list = glob.glob(fr"D:\intern\University-Release\train\drone\{index}\*.jpeg")
    for i in range(1):
        img_path = random.choice(img_list)
        base64_image = encode_image(img_path)
        user_message['content'].append(
            {
                "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
            }
        )
           
    user_message['content'].append( {"type": "text", "text": "describe the satellite view of the region in the image within 45 tokens unique for people to find the region on the map"})
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key='sk-c50916c57fbe41ffb70c09be373c9136',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max-2025-08-13", # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": "You are a helpful assistant."}]},
            user_message
            # {
            #     "role": "user",
            #     "content": [
            #         {
            #             "type": "image_url",
            #             # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
            #             # PNG图像：  f"data:image/png;base64,{base64_image}"
            #             # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
            #             # WEBP图像： f"data:image/webp;base64,{base64_image}"
            #             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
            #         },
            #         {"type": "text", "text": "describe the satellite view of the target building in the images' center for user to find them on the satellite image in few sentences"},
            #     ],
            # }
        ],
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    import os
    # for index in os.listdir(r'D:\intern\University-Release\test\gallery_drone'):
    for index in os.listdir(r'D:\intern\University-Release\train\drone'):
        output_path = fr'D:\intern\data\drone_text_sec\{index}.json'
        if os.path.exists(output_path):
            continue
        message = call_with_messages(index)

        try:
            output = json.loads(message)
            json.dump(output, open(output_path, 'w'))
        except:
            print('error: ',index)