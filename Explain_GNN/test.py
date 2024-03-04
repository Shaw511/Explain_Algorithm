import numpy as np
from globalBalancing_baseline import *
#
#
# X_in = np.loadtxt("D:\Pycharm\PycharmProjects\PGMExplainer-wyj\PGM_Node\Explain_GNN\X_in.txt", delimiter=',')
# Y_in = np.loadtxt("D:\Pycharm\PycharmProjects\PGMExplainer-wyj\PGM_Node\Explain_GNN\Y_in.txt", delimiter=',')
# Y_in= Y_in.reshape(-1, 1)
#
# print(X_in.shape)
# print(Y_in.shape)
# print(type(X_in))
# print(type(Y_in))
#
# learning_rate = 0.001;
# num_steps = 3000;
# tol = 1e-8
#
# RMSE_gb, F1_gb, W = f_baseline_globalBalancing(1, X_in, Y_in, learning_rate, num_steps, tol)
#
# print("当前的W_tensor是：", W)
# print(type(W))


# import numpy as np
#
# # 创建一个示例的NumPy数组
# arr = np.array([5, 2, 9, 1, 7, 3, 8, 4, 6])
#
# # 对数组进行降序排序并获取排序后的索引
# sorted_indices = np.argsort(arr)[::-1]  # [::-1]反转索引以获得降序排序
#
# # 取前五个元素的索引
# top_five_indices = sorted_indices[:5] + 1
#
# print(top_five_indices)

from collections import deque

# 示例的5维邻接矩阵，1表示节点相邻，0表示不相邻
adjacency_matrix = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0]
]


def shortest_path(adjacency_matrix, start_node, end_node):
    if start_node == end_node:
        return 0  # 起点和终点相同，距离为0

    n = len(adjacency_matrix)
    visited = [False] * n
    queue = deque([(start_node, 0)])

    while queue:
        current_node, distance = queue.popleft()
        visited[current_node] = True

        for neighbor, is_connected in enumerate(adjacency_matrix[current_node]):
            if is_connected and not visited[neighbor]:
                if neighbor == end_node:
                    return distance + 1  # 找到终点，返回距离
                queue.append((neighbor, distance + 1))

    return -1  # 无法到达终点


start_node = 0  # 起始节点
end_node = 3    # 目标节点

distance = shortest_path(adjacency_matrix, start_node, end_node)
if distance != -1:
    print(f"节点 {start_node} 到节点 {end_node} 的最短距离为 {distance} 跳")
else:
    print(f"节点 {start_node} 到节点 {end_node} 之间没有直接连接")
