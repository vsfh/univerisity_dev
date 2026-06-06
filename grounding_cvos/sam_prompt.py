import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize
from model.TROGeo import TROGeo
from util import *
from segment_anything import SamPredictor, sam_model_registry
import torch.nn.functional as F
from PIL import Image

# ======================bbox-point=========================
dataset_name = "CVOGL_DroneAerial"
pretrain = "saved_models/model_droneaerial_model_best.pth.tar"
save_dir = "results/TROGeo-bbox-point/CVOGL_DroneAerial"

# dataset_name = 'CVOGL_SVI'
# pretrain = 'saved_models/model_svi_model_best.pth.tar'
# save_dir = 'results/TROGeo-bbox-point/CVOGL_SVI'

if dataset_name == "CVOGL_SVI":
    query_featuremap_hw = [256, 512]
else:
    query_featuremap_hw = [256, 256]

root_dir = "/data/feihong/CVOGL"
root_dir = os.path.join(root_dir, dataset_name)
queryimg_dir = os.path.join(root_dir, "query")
rsimg_dir = os.path.join(root_dir, "satellite")

input_transform = Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

model = TROGeo()
model = torch.nn.DataParallel(model)
checkpoint = torch.load(pretrain, map_location="cpu")
pretrained_dict = checkpoint["state_dict"]
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
assert len([k for k, v in pretrained_dict.items()]) != 0
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

del checkpoint
torch.cuda.empty_cache()

model.eval()
for param in model.parameters():
    param.requires_grad = False
model = model.cuda()


# sam
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.eval()
for param in sam.parameters():
    param.requires_grad = False
sam.to(device=device)

for split_name in ["val", "test"]:
    # mIoU, mDice, AAE, ME = 0, 0, 0, 0
    data_path = os.path.join(root_dir, "{0}_{1}.pth".format(dataset_name, split_name))
    data_list = torch.load(data_path)

    for data in tqdm(data_list):
        _, queryimg_name, rsimg_name, _, click_xy, bbox, _, cls_name = data

        queryimg = cv2.imread(os.path.join(queryimg_dir, queryimg_name))
        queryimg = cv2.cvtColor(queryimg, cv2.COLOR_BGR2RGB)

        rsimg = cv2.imread(os.path.join(rsimg_dir, rsimg_name))
        rsimg = cv2.cvtColor(rsimg, cv2.COLOR_BGR2RGB)

        queryimg = input_transform(queryimg)
        rsimg = input_transform(rsimg)

        queryimg = queryimg.unsqueeze(0).cuda()
        rsimg = rsimg.unsqueeze(0).cuda()

        click_hw = (int(click_xy[1]), int(click_xy[0]))
        mat_clickhw = np.zeros(
            (query_featuremap_hw[0], query_featuremap_hw[1]), dtype=np.float32
        )
        click_h = [pow(one - click_hw[0], 2) for one in range(query_featuremap_hw[0])]
        click_w = [pow(one - click_hw[1], 2) for one in range(query_featuremap_hw[1])]
        norm_hw = pow(
            query_featuremap_hw[0] * query_featuremap_hw[0]
            + query_featuremap_hw[1] * query_featuremap_hw[1],
            0.5,
        )
        for i in range(query_featuremap_hw[0]):
            for j in range(query_featuremap_hw[1]):
                tmp_val = 1 - (pow(click_h[i] + click_w[j], 0.5) / norm_hw)
                mat_clickhw[i, j] = tmp_val * tmp_val

        mat_clickhw = torch.Tensor(mat_clickhw).unsqueeze(0).cuda()
        pred_anchor, pred_coords = model(queryimg, rsimg, mat_clickhw)
        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        pred_bbox = pred_anchor2bbox(pred_anchor)

        # SAM
        reference_embeddings = sam.image_encoder(rsimg)
        b, c, h, w = rsimg.shape
        prompt_points = (
            torch.cat(
                [
                    (pred_bbox[:, 0] + pred_bbox[:, 2]) / 2.0,
                    (pred_bbox[:, 1] + pred_bbox[:, 3]) / 2.0,
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        prompt_labels = torch.Tensor([1]).cuda().unsqueeze(0)
        # Embed prompts
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=(prompt_points, prompt_labels),
            boxes=pred_bbox,
            masks=None,
        )

        # Predict masks
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=reference_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = F.interpolate(
            low_res_masks, scale_factor=4, mode="bilinear", align_corners=False
        )

        masks = masks.cpu().numpy()
        masks[masks > 0.5] = 1
        masks[masks < 1] = 0
        masks = masks.astype(np.uint8) * 255
        masks = masks.squeeze()
        image = Image.fromarray(masks, mode="L")
        vis_rsimg_name = "{}--{}--bbox({},{},{},{}).jpg".format(
            queryimg_name[:-4],
            rsimg_name[:-4],
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[3]),
        )
        image.save(os.path.join(save_dir, split_name, vis_rsimg_name))
