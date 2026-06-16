import torch
import numpy as np

anchors = '37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573'
anchors_full = np.array([float(x.strip()) for x in anchors.split(',')])
anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()

def xywh2xyxy(x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    #y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y

def pred_anchor2bbox(pred_anchor):
    batch_size, grid_stride = 1, 16
    pred_confidence = pred_anchor[:, :, 4, :, :]
    scaled_anchors = anchors_full / grid_stride

    # pred_gi, pred_gj = torch.zeros_like(target_gi), torch.zeros_like(target_gj)
    # pred_bbox = torch.zeros_like(target_bbox)
    pred_bbox = torch.zeros([1, 4]).cuda()
    for batch_idx in range(batch_size):
        best_n, gj, gi = torch.where(pred_confidence[batch_idx].max() == pred_confidence[batch_idx])
        best_n, gj, gi = best_n[0], gj[0], gi[0]
        # pred_gj[batch_idx], pred_gi[batch_idx] = gj, gi
        # print((best_n, gi, gj))

        pred_bbox[batch_idx, 0] = pred_anchor[batch_idx, best_n, 0, gj, gi].sigmoid() + gi
        pred_bbox[batch_idx, 1] = pred_anchor[batch_idx, best_n, 1, gj, gi].sigmoid() + gj
        pred_bbox[batch_idx, 2] = torch.exp(pred_anchor[batch_idx, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
        pred_bbox[batch_idx, 3] = torch.exp(pred_anchor[batch_idx, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
    pred_bbox = pred_bbox * grid_stride
    pred_bbox = xywh2xyxy(pred_bbox)

    return pred_bbox

def find_centroids_batch(mask):
    mask = mask[:, 0, :, :]

    centroids = []
    for b in range(mask.size(0)):
        y_coords, x_coords = torch.where(mask[b] > 0)

        if len(x_coords) == 0 or len(y_coords) == 0:
            # centroids.append((None, None))
            centroids.append([0, 0])
        else:
            centroid_x = torch.mean(x_coords.float())
            centroid_y = torch.mean(y_coords.float())
            centroids.append([centroid_x.item(), centroid_y.item()])

    return centroids
