import numpy as np
from matplotlib.patches import Rectangle
import random_map
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d
import math
import json

from create_graph import build_graph
from dijkstra import dijkstra

# 添加边
def add_edge(graph, node1, node2, dist):
    if node1 not in graph:
        graph[node1] = {}
    graph[node1][node2] = dist

# 提取坐标绘图用
def extract_coordinates(adjacency_matrix):
    # 使用两个列表分别存储x和y坐标
    x_coords = []
    y_coords = []

    # 遍历字典的键（节点）和值（相邻节点字典）
    for node, edges in adjacency_matrix.items():
        # 如果当前节点不在坐标列表中，则添加它的坐标
        if (node[0], node[1]) not in zip(x_coords, y_coords):
            x_coords.append(node[0])
            y_coords.append(node[1])
        # 添加当前节点的所有相邻节点
        for neighbor in edges:
            # 如果相邻节点不在坐标列表中，则添加它的坐标
            if (neighbor[0], neighbor[1]) not in zip(x_coords, y_coords):
                x_coords.append(neighbor[0])
                y_coords.append(neighbor[1])

    # 返回坐标列表
    return x_coords, y_coords


# 计算两点之间的欧几里得距离
def euclidean_distance(p1, p2):

    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# 连接终点
def connect_endpoint_if_necessary(graph, end, dmin, dmax):
    # 检查终点是否作为值存在于图中的任何相邻节点列表中
    if end not in [neighbor for node in graph for neighbor in graph[node]]:
        # 收集所有x坐标在指定范围内的节点
        candidates = [neighbor for node in graph for neighbor in graph[node] if (dmin <= euclidean_distance(end, neighbor) <= dmax)]
        # 将终点与这些节点连接起来
        for candidate in candidates:
            # 终点作为新节点添加到图中
            add_edge(graph, candidate, end, euclidean_distance(end, candidate))  # 连接当前点与候选点（有向图）
            # print('adding edge', candidate, 'to', end)
    return graph

def modify_tuples(tuples_list, origin_s, origin_e):
    # 检查列表是否为空
    if not tuples_list:
        return []

    # 创建一个新的列表来存储结果
    modified_list = []

    # 替换第一个元组
    modified_list.append(origin_s)

    # 遍历中间的元组并进行修改
    for tuple in tuples_list[1:-1]:
        modified_tuple = (tuple[0] * 4, tuple[1]* 4)
        modified_list.append(modified_tuple)

    # 替换最后一个元组
    modified_list.append(origin_e)

    return modified_list


parser = argparse.ArgumentParser(description='argparse learning')  # 创建解析器
parser.add_argument('-id', type=int, help='envs id', default=102)  # 添加参数args = parser.parse_args() # 解析参数
args = parser.parse_args()  # 解析参数

print(f'-----------------args.id:{args.id}-----------------------')

# 初始化地图
map = random_map.RandomMap(args.id)
# plt.imshow(map.map)
# plt.show()

# 定义池化核的大小
pool_size = 5

# 计算输出数组的尺寸
output_height = map.map.shape[0] // pool_size
output_width = map.map.shape[1] // pool_size

# 初始化输出数组
pooled_array = np.zeros((output_height, output_width))

# 应用最大值池化
for i in range(output_height):
    for j in range(output_width):
        # 选择当前池化窗口
        window = map.map[i * pool_size:(i + 1) * pool_size, j * pool_size:(j + 1) * pool_size]
        # 计算窗口内的最大值，并赋值给输出数组
        pooled_array[i, j] = np.max(window)

# 打印池化后的数组形状
# show(pooled_array)
# print(pooled_array.shape)

start = (round(map.start[1] // pool_size), round(map.start[0] // pool_size))  # 起点
end = (round(map.end[1] // pool_size), round(map.end[0] // pool_size))  # 终点
dmin = 10  # 最小距离
dmax = 20  # 最大距离

graph = build_graph(pooled_array, start, end, dmin, dmax)

# 连接终点
graph = connect_endpoint_if_necessary(graph, end, dmin, dmax)

# 输出横纵坐标
x_nodes, y_nodes = extract_coordinates(graph)

# 使用matplotlib绘制数组
plt.scatter(y_nodes, x_nodes, color='blue')
plt.scatter(start[1], start[0], color='green', marker='o')
plt.scatter(end[1], end[0], color='red', marker='o')
plt.imshow(pooled_array)  # 使用灰度颜色图

if end in [neighbor for node in graph for neighbor in graph[node]]:
    # print('end is in graph')
    distance, path = dijkstra(graph, start, end)
    print(f"The shortest distance from {start} to {end} is {distance}.")
    print(f"The path is {path}.")
else:
    print('end not in graph')
    path = []


unsample_path = modify_tuples(path, (map.start[1], map.start[0]), (map.end[1], map.end[0]))

item = {}
item['index'] = args.id
item['coordinate'] = unsample_path

with open(f'DijkstraResult\\{args.id}.json', 'w') as json_file:
    json.dump(item, json_file, indent=4)

plt.show()
print(graph)

