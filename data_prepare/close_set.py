import os
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

from sklearn.cluster import DBSCAN
from geopy.distance import geodesic

def geodesic_distance_metric(coord1, coord2):
    """
    一个自定义的距离度量函数，用于DBSCAN。
    计算两个地理坐标点之间的大地距离（米）。
    
    Args:
        coord1: 第一个点的 (latitude, longitude) 元组。
        coord2: 第二个点的 (latitude, longitude) 元组。

    Returns:
        两点间的大地距离（米）。
    """
    return geodesic(coord1, coord2).m

def group_files_by_proximity(file_data: list[tuple], max_distance: int = 500) -> list[list]:
    """
    使用DBSCAN算法根据地理距离对文件进行分组。

    Args:
        file_data: 包含 (filename, lat, lon) 元组的列表。
        max_distance: 聚类的最大距离阈值（米）。

    Returns:
        一个列表的列表，每个子列表包含属于同一组的文件名。
    """
    if not file_data:
        return
        
    filenames = [item[0] for item in file_data]
    # 将坐标数据转换为DBSCAN所需的NumPy数组格式
    coords = np.array([(item[1], item[2]) for item in file_data])
    
    # 实例化DBSCAN算法，并设置关键参数
    # eps: 半径阈值，即用户需求的500米
    # min_samples: 一个簇所需的最小点数，设定为2确保每个簇至少包含两个点
    # metric: 指定自定义距离度量函数
    dbscan = DBSCAN(eps=max_distance, min_samples=2, metric=geodesic_distance_metric)
    
    # 执行聚类
    dbscan.fit(coords)
    labels = dbscan.labels_
    
    # 将聚类结果（标签）映射回原始文件名
    clusters = {}
    for filename, label in zip(filenames, labels):
        # -1 标签表示噪声（不属于任何簇）
        if label!= -1:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(filename)
            
    # 将字典的值转换为所需的列表的列表格式
    return list(clusters.values())

def parse_kml_files(directory_path: str) -> list[tuple]:
    """
    遍历指定目录下的KML文件，并提取每个文件中的第一个Placemark坐标。

    Args:
        directory_path: KML文件所在的目录路径。

    Returns:
        一个包含 (filename, latitude, longitude) 元组的列表。
    """
    # 使用 pathlib 进行高效、跨平台的文件遍历
    directory = Path(directory_path)
    file_data = []
    for file_path in directory.glob('*.kml'):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # 定义命名空间（不同KML文件可能使用不同的命名空间）
            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            
            # 修改name元素为文件名
            latitude = root.find('.//kml:latitude', ns).text
            longitude = root.find('.//kml:longitude', ns).text

            file_data.append((file_path.name, float(latitude), float(longitude)))

        except Exception as e:
            # 健壮性设计：处理解析错误或意外文件结构
            print(f"Error parsing file {file_path.name}: {e}")
            continue
    return file_data
if __name__=='__main__':
    import json
    a = group_files_by_proximity(parse_kml_files(r'D:\intern\data\kml_512'), 200)
    output = {}
    for i,b in enumerate(a):
        if len(b)>1:
            output[i] = b
    json.dump(output, open('b.json','w'))