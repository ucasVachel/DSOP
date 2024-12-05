import numpy as np
import math


def is_closer_to_start_than_end(p, start, end):
    """判断候选点到终点的距离是否小于候选点到终点的距离"""
    dist_start = np.linalg.norm(np.array(start) - np.array(end))
    dist_end = np.linalg.norm(np.array(p) - np.array(end))
    return dist_start > dist_end


def is_clear_path(start, candidate, np_map):
    """检查从start到candidate之间的路径是否清晰（即没有值为2的点）"""
    x0, y0 = start
    x1, y1 = candidate
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        # 检查当前点是否为2（不可通过的障碍）
        if np_map[round(x0), round(y0)] == 2:
            return False
        if round(x0) == x1 and round(y0) == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return True

def generate_candidate_points(start, end, dmin, dmax, angle_range_deg=50):
    """根据起点和终点生成候选点（指向终点的小扇形区域）"""
    candidates = set()

    # 计算从起点到终点的方向角度
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    target_angle = math.atan2(delta_y, delta_x)  # 起点到终点的角度（弧度）

    # 将角度范围从度转换为弧度
    angle_range_rad = math.radians(angle_range_deg)

    # 在 dmin 到 dmax 之间生成多个半径值
    num_points = 3  # 每个圆周上生成的点数，可以根据需要调整
    for r in np.linspace(dmin, dmax, num_points):  # 半径范围在 [dmin, dmax]
        for angle in np.linspace(target_angle - angle_range_rad, target_angle + angle_range_rad, num_points):
            x = round(start[0] + r * np.cos(angle))  # 计算点的x坐标并四舍五入
            y = round(start[1] + r * np.sin(angle))  # 计算点的y坐标并四舍五入
            candidates.add((x, y))

    return list(candidates)


def build_graph(np_map, start, end, dmin, dmax):
    """根据numpy地图构建图（有向图）"""
    rows, cols = np_map.shape
    graph = {}

    def add_edge(node1, node2, dist):
        if node1 not in graph:
            graph[node1] = {}
        graph[node1][node2] = dist

    # 初始化：起点作为第一个搜寻点
    current_search_points = [start]

    # 迭代过程：不断扩展搜寻点集
    while current_search_points:
        next_search_points = []

        for point in current_search_points:
            # 生成当前点的候选点（只在指向终点的小扇形区域内）
            candidates = generate_candidate_points(point, end, dmin, dmax)

            for candidate in candidates:
                x, y = candidate
                if 0 <= x < rows and 0 <= y < cols:  # 保证候选点在地图范围内
                    if np_map[x, y] == 0 and is_closer_to_start_than_end(candidate, point, end) and is_clear_path(point, candidate, np_map):  # 只有值为0且未访问的点才能作为节点
                        dist = np.linalg.norm(np.array(point) - np.array(candidate))
                        add_edge(point, candidate, dist)  # 连接当前点与候选点（有向图）
                        next_search_points.append(candidate)  # 将候选点加入下一轮搜寻点
        # 更新搜寻点集
        current_search_points = next_search_points

        #print(graph)

    return graph

# 示例使用
# np_map = np.array([
#     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 2, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 2, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#     [0, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ])
#
# start = (0, 0)  # 起点
# end = (12, 12)  # 终点
# dmin = 3.7  # 最小距离
# dmax = 4.1  # 最大距离
#
# graph = build_graph(np_map, start, end, dmin, dmax)
#
# print(graph)
