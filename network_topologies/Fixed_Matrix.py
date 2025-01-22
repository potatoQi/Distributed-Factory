import networkx as nx
import numpy as np
import random

class Fixed_Matrix:
    def __init__(self, number_of_nodes, graph_type, matrix_type, num_graphs=1, density=0.2):
        self.number_of_nodes = number_of_nodes
        self.graph_type = graph_type    # 链/菊花/二叉树
        self.matrix_type = matrix_type  # 行随机/列随机/双随机
        self.density = density          # random图的密度

    def generate_topology(self):
        graph = self._generate_strongly_connected_graph()
        adjacency_matrix = nx.adjacency_matrix(graph).todense()
        if self.matrix_type == 'row_stochastic':
            stochastic_matrix = self._make_row_stochastic(adjacency_matrix.T)
        elif self.matrix_type == 'column_stochastic':
            stochastic_matrix = self._make_column_stochastic(adjacency_matrix.T)
        elif self.matrix_type == 'double_stochastic':
            graph, stochastic_matrix = self._generate_double_stochastic(adjacency_matrix.T)
        else:
            raise ValueError(f"Unknown matrix type: {self.matrix_type}")
        return graph, stochastic_matrix

    def _generate_strongly_connected_graph(self):
        while True:
            graph = self._generate_graph()
            if nx.is_strongly_connected(graph):
                return graph

    def _generate_graph(self):
        if self.graph_type == "chain":
            graph = nx.path_graph(self.number_of_nodes, create_using=nx.DiGraph())
            # 手动添加回边以确保强连通性
            for i in range(1, self.number_of_nodes):
                graph.add_edge(i, i-1)
        elif self.graph_type == "star":
            graph = nx.DiGraph()
            center_node = 0
            for i in range(1, self.number_of_nodes):
                graph.add_edge(center_node, i)  # 中心节点到叶子节点的有向边
                graph.add_edge(i, center_node)  # 叶子节点到中心节点的反向边
        elif self.graph_type == "random_binary_tree":
            graph = nx.random_tree(self.number_of_nodes, create_using=nx.DiGraph())
            for u, v in list(graph.edges()):
                graph.add_edge(v, u)
        elif self.graph_type == "perfect_binary_tree":
            k = int(np.log2(self.number_of_nodes + 1))
            closest_number = 2**k - 1
            if closest_number != self.number_of_nodes:
                raise ValueError(f"输入的节点数 {self.number_of_nodes} 不能构成一个完美二叉树。"
                                f"最近的能构成完美二叉树的节点数是 {closest_number}。")
            graph = nx.balanced_tree(2, k - 1, create_using=nx.DiGraph())
            for u, v in list(graph.edges()):
                graph.add_edge(v, u)
        elif self.graph_type == "random_graph":
            graph = nx.erdos_renyi_graph(self.number_of_nodes, self.density, directed=True)  # 设置为0.2以减小边的密度
        elif self.graph_type == "full":
            graph = nx.erdos_renyi_graph(self.number_of_nodes, 1, directed=True)
        elif self.graph_type == "cycle":
            graph = nx.DiGraph()
            for i in range(self.number_of_nodes):
                graph.add_edge(i, (i + 1) % self.number_of_nodes)
        elif self.graph_type == "spider_web":
            # 自动计算层数和每层的节点数
            layers = int(np.sqrt(self.number_of_nodes))  # 尝试使用平方根作为层数
            nodes_per_layer = self.number_of_nodes // layers  # 每层的平均节点数
            # 检查是否能构成一个合理的蜘蛛网
            if layers * nodes_per_layer != self.number_of_nodes:
                # 如果不能均匀分配，计算最近的有效节点数
                closest_number = layers * nodes_per_layer
                raise ValueError(f"输入的节点数 {self.number_of_nodes} 不能构成一个合理的蜘蛛网。"
                                f"最近的有效节点数是 {closest_number}, "
                                f"或尝试使用 {layers * (nodes_per_layer + 1)}")
            graph = nx.DiGraph()
            # 添加每层的节点并创建环
            for layer in range(layers):
                start_index = layer * nodes_per_layer
                end_index = start_index + nodes_per_layer
                layer_nodes = list(range(start_index, end_index))
                # 创建每层的环
                for i in range(len(layer_nodes)):
                    graph.add_edge(layer_nodes[i], layer_nodes[(i + 1) % len(layer_nodes)])
                    # 随机决定是否为相邻节点添加双向边
                    if random.random() > 0.5:  # 控制双向边的概率
                        graph.add_edge(layer_nodes[(i + 1) % len(layer_nodes)], layer_nodes[i])
                # 连接相邻层的节点，添加双向边
                if layer > 0:
                    previous_layer_nodes = list(range((layer - 1) * nodes_per_layer, layer * nodes_per_layer))
                    for i in range(len(layer_nodes)):
                        graph.add_edge(previous_layer_nodes[i], layer_nodes[i])
                        graph.add_edge(layer_nodes[i], previous_layer_nodes[i])
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
        return graph

    def _make_row_stochastic(self, matrix):
        np.fill_diagonal(matrix, 1)  # 确保对角线元素为 1
        row_sums = matrix.sum(axis=1)
        stochastic_matrix = np.zeros_like(matrix, dtype=float)
        for i in range(matrix.shape[0]):
            if row_sums[i] > 0:
                stochastic_matrix[i] = matrix[i] / row_sums[i]
        return stochastic_matrix

    def _make_column_stochastic(self, matrix):
        np.fill_diagonal(matrix, 1)  # 确保对角线元素为 1
        col_sums = matrix.sum(axis=0)
        stochastic_matrix = np.zeros_like(matrix, dtype=float)
        for j in range(matrix.shape[1]):
            if col_sums[j] > 0:
                stochastic_matrix[:, j] = matrix[:, j] / col_sums[j]
        return stochastic_matrix

    def _make_double_stochastic(self, matrix):
        # 确保每行的非零元素数量相同并且非零元素值相同
        row_nonzero_count = np.count_nonzero(matrix[0, :])
        col_nonzero_count = np.count_nonzero(matrix[:, 0])
        if row_nonzero_count != col_nonzero_count:
            raise ValueError("行和列的非零元素数量不相等，无法生成双随机矩阵")
        row_value = 1.0 / row_nonzero_count
        col_value = 1.0 / col_nonzero_count
        double_stochastic_matrix = np.zeros_like(matrix, dtype=float)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0:
                    double_stochastic_matrix[i, j] = row_value
        return double_stochastic_matrix

    def _generate_double_stochastic(self, matrix):
        while True:
            try:
                np.fill_diagonal(matrix, 1)  # 确保对角线元素为 1
                # 检查每行和每列的非零元素数量是否一致
                row_nonzero_counts = np.count_nonzero(matrix, axis=1)
                col_nonzero_counts = np.count_nonzero(matrix, axis=0)
                if not np.all(row_nonzero_counts == row_nonzero_counts[0]) or not np.all(col_nonzero_counts == col_nonzero_counts[0]):
                    # 如果不一致，重新生成图
                    graph = self._generate_strongly_connected_graph()
                    matrix = nx.adjacency_matrix(graph).todense().T
                    continue
                # 直接生成双随机矩阵
                double_stochastic_matrix = self._make_double_stochastic(matrix)
                return graph, double_stochastic_matrix
            except ValueError:
                # 如果生成失败，继续尝试
                graph = self._generate_strongly_connected_graph()
                matrix = nx.adjacency_matrix(graph).todense().T