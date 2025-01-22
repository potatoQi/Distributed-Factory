import networkx as nx
import numpy as np
import random

class TimeVarying_Matrix:
    def __init__(self, number_of_nodes, graph_type, matrix_type, num_graphs=3, density=0.2):
        self.number_of_nodes = number_of_nodes
        self.graph_type = graph_type    # 链/菊花/随机图/循环/蜘蛛网
        self.matrix_type = matrix_type  # 行随机/列随机/双随机
        self.num_graphs = num_graphs    # 生成的图数量
        self.density = density          # random图的密度

    def generate_topology(self):
        while True:  # 循环直到找到一个强连通的叠加结果
            graphs = []
            stochastic_matrices = []

            for _ in range(self.num_graphs):
                graph = self._generate_not_strongly_connected_graph()
                adjacency_matrix = nx.adjacency_matrix(graph).todense()
                if self.matrix_type == 'row_stochastic':
                    stochastic_matrix = self._make_row_stochastic(adjacency_matrix.T)
                elif self.matrix_type == 'column_stochastic':
                    stochastic_matrix = self._make_column_stochastic(adjacency_matrix.T)
                elif self.matrix_type == 'double_stochastic':
                    graph, stochastic_matrix = self._generate_double_stochastic(adjacency_matrix.T)
                else:
                    raise ValueError(f"Unknown matrix type: {self.matrix_type}")

                graphs.append(graph)
                stochastic_matrices.append(stochastic_matrix)

            # 将指定数量的图组合成一个强连通图，并生成相应的矩阵
            combined_graph, combined_stochastic_matrix = self._combine_graphs(graphs)
            if nx.is_strongly_connected(combined_graph):
                # 返回单个图列表、各自的矩阵、组合后的强连通图及其矩阵
                return graphs, stochastic_matrices, combined_graph, combined_stochastic_matrix

    def _generate_not_strongly_connected_graph(self):
        # 生成不强连通的图
        while True:
            graph = self._generate_graph()
            if not nx.is_strongly_connected(graph):
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

    def _combine_graphs(self, graphs):
        # 将多个图的边集合并，生成一个强连通图
        combined_graph = nx.DiGraph()
        for graph in graphs:
            combined_graph.add_edges_from(graph.edges())
        combined_adjacency_matrix = nx.adjacency_matrix(combined_graph).todense()
        # 获取节点顺序
        node_order = list(combined_graph.nodes())
        # 按自然顺序重排邻接矩阵
        sorted_indices = sorted(range(len(node_order)), key=lambda k: node_order[k])
        combined_adjacency_matrix = combined_adjacency_matrix[sorted_indices, :][:, sorted_indices]

        # 根据指定的矩阵类型生成叠加图的矩阵
        if self.matrix_type == 'row_stochastic':
            combined_stochastic_matrix = self._make_row_stochastic(combined_adjacency_matrix.T)
        elif self.matrix_type == 'column_stochastic':
            combined_stochastic_matrix = self._make_column_stochastic(combined_adjacency_matrix.T)
        elif self.matrix_type == 'double_stochastic':
            np.fill_diagonal(combined_adjacency_matrix.T, 1)
            if np.all(np.count_nonzero(combined_adjacency_matrix.T, axis=1) == np.count_nonzero(combined_adjacency_matrix.T, axis=1)[0]) and \
               np.all(np.count_nonzero(combined_adjacency_matrix.T, axis=0) == np.count_nonzero(combined_adjacency_matrix.T, axis=0)[0]):
                combined_stochastic_matrix = self._make_double_stochastic(combined_adjacency_matrix.T)
            else:
                # 返回最简的非强连通图
                non_strongly_connected_graph = nx.DiGraph()
                non_strongly_connected_graph.add_edges_from([(0, 1)])
                return non_strongly_connected_graph, None
        else:
            raise ValueError(f"Unknown matrix type: {self.matrix_type}")
        return combined_graph, combined_stochastic_matrix

    def _make_row_stochastic(self, matrix):
        np.fill_diagonal(matrix, 1)
        row_sums = matrix.sum(axis=1)
        stochastic_matrix = np.zeros_like(matrix, dtype=float)
        for i in range(matrix.shape[0]):
            if row_sums[i] > 0:
                stochastic_matrix[i] = matrix[i] / row_sums[i]
        return stochastic_matrix

    def _make_column_stochastic(self, matrix):
        np.fill_diagonal(matrix, 1)
        col_sums = matrix.sum(axis=0)
        stochastic_matrix = np.zeros_like(matrix, dtype=float)
        for j in range(matrix.shape[1]):
            if col_sums[j] > 0:
                stochastic_matrix[:, j] = matrix[:, j] / col_sums[j]
        return stochastic_matrix

    def _make_double_stochastic(self, matrix):
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
                np.fill_diagonal(matrix, 1)
                row_nonzero_counts = np.count_nonzero(matrix, axis=1)
                col_nonzero_counts = np.count_nonzero(matrix, axis=0)
                if matrix == np.eye(len(matrix)) or not np.all(row_nonzero_counts == row_nonzero_counts[0]) or not np.all(col_nonzero_counts == col_nonzero_counts[0]):
                    graph = self._generate_not_strongly_connected_graph()
                    matrix = nx.adjacency_matrix(graph).todense().T
                    continue
                double_stochastic_matrix = self._make_double_stochastic(matrix)
                return graph, double_stochastic_matrix
            except ValueError:
                graph = self._generate_not_strongly_connected_graph()
                matrix = nx.adjacency_matrix(graph).todense().T
