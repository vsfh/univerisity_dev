# Breaking Rectangular Shackles: Cross-View Object Segmentation for Fine-Grained Object Geo-Localization

<p align="center">
    <img src="images/logo.jpg"/ style="width:60%; height:auto; display:inline-block;">
<p>

<p align="center">
        📄<a href="https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Breaking_Rectangular_Shackles_Cross-View_Object_Segmentation_for_Fine-Grained_Object_Geo-Localization_ICCV_2025_paper.html">Paper</a> &nbsp ｜ 💻<a href="https://zqwlearning.github.io/CVOS/">Project</a> &nbsp ｜ &nbsp 📊<a href="https://drive.google.com/file/d/1b3w0uY16uhkYI80d14pNMS2E4_ml7CAV/view?usp=sharing">Dataset</a> ｜ &nbsp 🖼️<a href="https://iccv.thecvf.com/media/PosterPDFs/ICCV%202025/570.png?t=1760008578.6223292">Poster</a>
</p>

<div style="text-align: center;">
    <p style="font-weight: bold; font-size: 1.2em;">
        📢🎉 It has been accepted by ICCV-25 🎉📢
    </p>
    <img src="images/iccv-logo.svg" style="width:300px; height:auto; display:inline-block;">
</div>
<div style="text-align: center; font-weight: bold; font-size: 1.1em;">
    This is a PyTorch implementation of the “Breaking Rectangular Shackles: Cross-View Object Segmentation for Fine-Grained Object Geo-Localization”.
</div>

# Dataset

The CVOGL-Seg dataset is created based on the CVOGL dataset; please download [CVOGL](https://drive.google.com/file/d/1WCwnK_rrU--ZOIQtmaKdR0TXcmtzU4cf/view?usp=sharing) first.

Our CVOGL-Seg dataset only provides the object segmentation mask; please download it from the [link](https://drive.google.com/file/d/1b3w0uY16uhkYI80d14pNMS2E4_ml7CAV/view?usp=sharing).

Download CVOGL and CVOGL-Seg and place them in the same directory, such as "/data/feihong/".

# Usage

## Train

Train on the **Drone → Satellite** task.

```shell
bash run_train_droneaerial.sh
```

```python
python train.py --emb_size 768 --img_size 1024 --max_epoch 25 --data_root /data/feihong/CVOGL --seg_root /data/feihong/CVOGL-Seg --data_name CVOGL_DroneAerial --beta 1.0 --savename model_droneaerial --gpu 0,1 --batch_size 12 --num_workers 24 --print_freq 50
```

Train on the **Ground → Satellite** task.

```shell
bash run_train_svi.sh
```

```python
python train.py --emb_size 768 --img_size 1024 --max_epoch 25 --data_root /data/feihong/CVOGL --seg_root /data/feihong/CVOGL-Seg --data_name CVOGL_SVI --beta 1.0 --savename model_svi --gpu 0,1 --batch_size 12 --num_workers 8 --print_freq 50
```

## Evaluation

Evaluate on the **Drone → Satellite** task.

```shell
bash run_test_droneaerial.sh
```

```python
python train.py --val --pretrain saved_models/model_droneaerial_model_best.pth.tar --emb_size 768 --img_size 1024 --data_root /data/feihong/CVOGL --seg_root /data/feihong/CVOGL-Seg --data_name CVOGL_DroneAerial --savename test_model_droneaerial --gpu 0,1 --batch_size 12 --num_workers 16 --print_freq 50

python train.py --test --pretrain saved_models/model_droneaerial_model_best.pth.tar --emb_size 768 --img_size 1024 --data_root /data/feihong/CVOGL --seg_root /data/feihong/CVOGL-Seg --data_name CVOGL_DroneAerial --savename test_model_droneaerial --gpu 0,1 --batch_size 12 --num_workers 16 --print_freq 50
```

Evaluate on the **Ground → Satellite** task.

```shell
bash run_test_svi.sh
```

```python
python train.py --val --pretrain saved_models/model_svi_model_best.pth.tar --emb_size 768 --img_size 1024 --data_root /data/feihong/CVOGL --seg_root /data/feihong/CVOGL-Seg --data_name CVOGL_SVI --savename test_model_svi --gpu 0,1 --batch_size 12 --num_workers 16 --print_freq 50

python train.py --test --pretrain saved_models/model_svi_model_best.pth.tar --emb_size 768 --img_size 1024 --data_root /data/feihong/CVOGL --seg_root /data/feihong/CVOGL-Seg --data_name CVOGL_SVI --savename test_model_svi --gpu 0,1 --batch_size 12 --num_workers 16 --print_freq 50
```

# SAM Prompt

Download the weight file from the link: [sam_vit_h_4b8939.pth](https://github.com/facebookresearch/segment-anything)。

```python
python sam_prompt.py
```

# Model Zoo

| Task                 | Download Link                                                |
| -------------------- | ------------------------------------------------------------ |
| Drone *→* Satellite  | [link](https://drive.google.com/file/d/1YlWTVGiWNGEEb0b_4rU6RIqSQhLg5FX1/view?usp=sharing) |
| Ground *→* Satellite | [link](https://drive.google.com/file/d/1MDEpopjDWDpfbDd2Co0osCoWVAcEiGeN/view?usp=sharing) |

# Acknowledgement

Our project makes extensive use of the codebases from **[DetGeo](https://github.com/sunyuxi/DetGeo)** and **[SAM](https://github.com/facebookresearch/segment-anything)**, and we express our sincere gratitude for their valuable contributions to the community. The **CVOGL-Seg** dataset was developed based on the **CVOGL** dataset, and we greatly appreciate the authors for making their dataset publicly available.

We would like to extend special thanks to **[Dr. Sun](https://orcid.org/0000-0002-3040-5880)**, who kindly provided detailed clarifications regarding the **GPS information** in the CVOGL dataset when we encountered related questions during the creation of CVOGL-Seg. The translated version of his email is included below (with anonymization applied) to facilitate understanding of the CVOGL dataset.

```sheel
The filenames of the satellite images in the dataset (the third field in each data record) contain coordinate information and the spatial scale. However, these coordinates are expressed in the Universal Transverse Mercator (UTM) projection system rather than GPS latitude and longitude coordinates, as the primary goal at the time was to facilitate the calculation of distances between pixels.

For example, consider the satellite image named 547433_5280574_512_32610.jpg:

1. 512 — This indicates that the image covers a ground area of 512 meters × 512 meters. Given that the image dimensions are 1024 × 1024 pixels, the spatial resolution (or scale) can be calculated as 512 meters / 1024 pixels = 0.5 meters per pixel.

2. (547433, 5280574) — These values represent the coordinates of the image center in meters. Based on the spatial resolution, the coordinates of any pixel within the image can be derived.

3. 32610 — This is the EPSG code corresponding to the UTM projection zone. Different regions have different EPSG codes. Specifically, the EPSG code for geographic coordinates (latitude and longitude) is EPSG:4326. Various open-source Python tools can be used to convert UTM coordinates into geographic coordinates.

4. The OpenStreetMap (OSM) details were not preserved in the dataset. However, you can retrieve detailed geographic information corresponding to these coordinates directly from OSM. OpenStreetMap is an open-source geographic information database that can be downloaded online and parsed using multiple open-source Python libraries.
```

 **Please cite their work when appropriate.**

```
@ARTICLE{sun10001754,
  title={Cross-view Object Geo-localization in a Local Region with Satellite Imagery}, 
  author={Yuxi Sun, Yunming Ye, Jian Kang, Ruben Fernandez-Beltran, Shanshan Feng, Xutao Li, Chuyao Luo, Puzhao Zhang, and Antonio Plaza},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  year={2023}
  doi={10.1109/TGRS.2023.3307508}
}
```

# Citation

```
@InProceedings{Zhang_2025_ICCV,
    author    = {Zhang, Qingwang and Zhu, Yingying},
    title     = {Breaking Rectangular Shackles: Cross-View Object Segmentation for Fine-Grained Object Geo-Localization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {8197-8206}
}
```



