import numpy as np
import os
import json
import matplotlib.pyplot as plt
import math

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def DTWDistance(s1, s2):
    DTW = {}

    # 构造D矩阵
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    # 动态规划打表
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = calculate_distance(s1[i], s2[j])
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])


# 指定包含JSON文件的文件夹路径
folder_path = 'RRTStar/rrtStarResult'
avg = 0
count = 0

with open('target.json') as file:
    # 将JSON数据加载到Python对象中
    annotations = json.load(file)

for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为json
    if filename.endswith('.json'):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)

        # 打开并读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            # 将JSON文件的内容解析为Python对象
            data = json.load(f)

            # 打印文件名和读取的数据，或者进行其他处理
            index = data['index']
            coordinate = data['coordinate']

            target = annotations[index]['coordinate']
            distance = DTWDistance(target, coordinate)

            avg += distance
            count += 1
            print(f'file_path:{file_path}  distance: {distance}')


print(f'avg: {avg/count}')
