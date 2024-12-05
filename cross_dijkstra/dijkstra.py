import heapq


def dijkstra(graph, start, end):
    # 构建所有节点的集合
    all_nodes = set()
    for node in graph:
        all_nodes.add(node)
        for neighbor in graph[node]:
            all_nodes.add(neighbor)

    # 用来保存从起点到任意点的距离
    distances = {vertex: float('infinity') for vertex in all_nodes}
    # 用来保存从起点到任意点的路径
    paths = {vertex: [] for vertex in all_nodes}
    # 起点到自身的距离是0
    distances[start] = 0
    # 路径中包含起点自身
    paths[start] = [start]

    # 优先队列，存储（距离，节点）
    pq = [(0, start)]

    while pq:
        # 取出队列中距离最小的节点
        current_distance, current_vertex = heapq.heappop(pq)

        # 如果当前距离大于已记录的距离，跳过当前节点
        if current_distance > distances[current_vertex]:
            continue

        # 有可能没有下一个可以到达的节点 毕竟单项
        if current_vertex not in graph:
            continue

        # 遍历当前节点的所有邻居
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            # 只有当找到更短的路径时才进行更新
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_vertex] + [neighbor]
                heapq.heappush(pq, (distance, neighbor))

    # 检查终点是否存在于图中
    if end not in distances:
        return "Destination not reachable", []

    # 只返回到终点的距离和路径
    return distances[end], paths[end]


# 示例图
# graph = {
#     'A': {'B': 1, 'C': 4},
#     'B': {'A': 1, 'C': 2, 'D': 5},
#     'C': {'A': 4, 'B': 2, 'D': 1},
#     'D': {'B': 5, 'C': 1}
# }
#
# start_vertex = 'A'
# end_vertex = 'D'
#
# # 计算最短距离和路径
# distance, path = dijkstra(graph, start_vertex, end_vertex)
# print(f"The shortest distance from {start_vertex} to {end_vertex} is {distance}.")
# print(f"The path is {path}.")