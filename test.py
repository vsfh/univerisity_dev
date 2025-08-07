from functional_cat.funcs.glip import GLIP
from PIL import Image
glip = GLIP(model_name="GLIP-L")
img_path = r'D:\intern\data\capture_kml\0000.jpg'
img = Image.open(img_path)
img = img.resize((img.width // 2, img.height // 2))
dets = glip([img], score_thres=0.5, class_labels=["black cat", "orange cat", "grey cat"])[0]
glip.draw_output_on_img(img, dets)