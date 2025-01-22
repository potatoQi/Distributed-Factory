import yaml
import matplotlib.pyplot as plt
import networkx as nx
from algorithms import DAGT_Algorithm, DACS_Algorithm, DACS_HB_Algorithm, MDGT_Algorithm
from network_topologies import Fixed_Matrix, TimeVarying_Matrix
from visualizations import plot_results
import numpy as np
import sys

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def spider_web_layout(layers, nodes_per_layer):
    pos = {}
    radius_increment = 2  # 每层之间的半径增量
    for layer in range(layers):
        radius = radius_increment * (layer + 1)
        theta = np.linspace(0, 2 * np.pi, nodes_per_layer, endpoint=False)
        layer_nodes = list(range(layer * nodes_per_layer, (layer + 1) * nodes_per_layer))
        for i, node in enumerate(layer_nodes):
            pos[node] = (radius * np.cos(theta[i]), radius * np.sin(theta[i]))
    return pos

def visualize_graph(graphs, graph_type):
    for graph in graphs:
        num_nodes = len(graph.nodes)  # 获取节点数量
        if graph_type == "spider_web":
            layers = int(np.sqrt(num_nodes))  # 自动计算层数
            nodes_per_layer = num_nodes // layers
            # 动态计算节点大小，节点越多，节点越小
            base_node_size = 1000  # 基础节点大小
            node_size = max(base_node_size / np.sqrt(num_nodes), 50)  # 确保节点不会太小
            # 使用自定义蜘蛛网布局
            pos = spider_web_layout(layers, nodes_per_layer)
            # 绘制图形
            plt.figure(figsize=(8, 8))
            nx.draw(graph, pos, with_labels=True, node_size=node_size, node_color='skyblue', edge_color='gray', font_size=10, font_weight='bold')
            plt.show()
        else:
            # 对于非spider_web的图类型，使用默认布局
            plt.figure(figsize=(8, 8))
            num_nodes = len(graph.nodes)
            # 根据图类型选择布局
            if graph_type == "chain":
                pos = nx.kamada_kawai_layout(graph)
            elif graph_type == "cycle":
                pos = nx.circular_layout(graph)
            else:
                pos = nx.spring_layout(graph)
            # 动态调整节点大小和边宽度
            node_size = max(1000 / np.sqrt(num_nodes), 100)  # 节点数多时节点变小，最小为100
            edge_width = max(2 / np.sqrt(num_nodes), 0.1)    # 节点数多时边变细，最细为0.1
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=node_size,
                node_color='skyblue',
                edge_color='gray',
                font_size=10,
                font_weight='bold',
                width=edge_width  # 动态调整边的宽度
            )
            plt.show()

def generate_doubly_stochastic_matrices(n):
    matrices = []
    for i in range(n):
        matrix = np.ones((n, n)) / (n - 1)  # 初始化为全联接，每个点1/(n-1)
        matrix[:, i] = 0  # 断开i-th节点的所有连接
        matrix[i, :] = 0  # 断开i-th节点的所有连接
        matrix[i, i] = 1  # 自环
        matrices.append(matrix)
    return matrices

def main(config):
    results = []
    
    for algorithm_config in config['algorithms']:
        algorithm_name = algorithm_config['name']
        
        # 获取每个算法的自定义参数
        network_topology = algorithm_config.get('network_topology')
        num_graphs = algorithm_config.get('num_graphs')
        number_of_nodes = algorithm_config.get('number_of_nodes')
        matrix_type = algorithm_config.get('matrix_type')
        graph_type = algorithm_config.get('graph_type')
        legend_name = algorithm_config.get('algorithm_name')
        density = algorithm_config.get('density')
        
        # 动态加载网络拓扑类
        topology_module = globals().get(network_topology, None)
        if topology_module is None:
            raise ValueError(f"Network topology '{network_topology}' not found.")
        topology_class = getattr(topology_module, network_topology)
        topology = topology_class(number_of_nodes, graph_type, matrix_type, num_graphs, density)

        graphs = []
        matrices = []
        if network_topology == "TimeVarying_Matrix" and matrix_type == "double_stochastic":
            matrices = generate_doubly_stochastic_matrices(5)
        elif network_topology == "Fixed_Matrix":
            graph, stochastic_matrix = topology.generate_topology()
            matrices.append(stochastic_matrix)
            graphs.append(graph)
        else:
            graphs, stochastic_matrices, combined_graph, combined_stochastic_matrix = topology.generate_topology()
            matrices = stochastic_matrices
            graphs.append(combined_graph)

        # 可视化每个算法的图
        print(f"Visualizing graph for algorithm {algorithm_name} with graph type {graph_type} and network topology {network_topology}")
        # visualize_graph(graphs, graph_type)  # 可视化时变图，或静态图
        # print(matrices)

        # 动态加载算法类
        algorithm_class = globals().get(algorithm_name)
        algorithm_class = getattr(algorithm_class, algorithm_name)
        if algorithm_class is None:
            raise ValueError(f"Algorithm '{algorithm_name}' not found.")
        
        # 实例化算法并传入配置参数
        algorithm = algorithm_class(algorithm_config['params'])
        
        # 运行算法
        result = algorithm.run(matrices)
        results.append((legend_name, result))

    # 比较和可视化所有算法的结果
    plot_results.plot_results(results)
    print('评测完成')

if __name__ == "__main__":
    config = load_config("config/default_config.yaml")
    main(config)
