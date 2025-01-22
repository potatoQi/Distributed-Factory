import matplotlib.pyplot as plt
import numpy as np

def plot_results(results):
    # 设置画布尺寸
    plt.rcParams['figure.figsize'] = (10, 6)

    # 定义标记符号和颜色
    markers = ['s', 'o', 'd', '^', '*', 'p', 'h', 'H', 'v', '<', '>', '8', 'x', 'X', 'D']
    colors = [
        'black', 'orangered', 'saddlebrown', 'blue', 'green', 'purple', 'magenta', 
        'cyan', 'gold', 'darkorange', 'pink', 'lightgreen', 'teal', 'navy', 'maroon'
    ]

    # 确保 markers 和 colors 的数量足够
    num_results = len(results)
    if num_results > len(markers):
        markers *= (num_results // len(markers)) + 1
    if num_results > len(colors):
        colors *= (num_results // len(colors)) + 1

    # 绘制每个结果
    for idx, (name, result) in enumerate(results):
        Iter_Num = len(result)
        plt.plot(range(1, Iter_Num + 1), result, 
                 label=name, linestyle='-', linewidth=1.5, markersize=6, 
                 markevery=range(Iter_Num // 10, Iter_Num, Iter_Num // 10), 
                 marker=markers[idx], color=colors[idx])

    # 设置x/y坐标标签及字体
    plt.xlabel('Iteration', fontsize=16, fontname='Times New Roman')
    plt.ylabel(r'$\|x_k - x^*\|$', fontsize=16, fontname='Times New Roman')

    # 将y轴刻度转换为10的负次方形式（对数刻度）
    plt.yscale('log')
    plt.yticks(fontsize=14)

    # 设置x轴刻度范围和间隔，并确保从0开始
    plt.xlim(0, max(len(result) for _, result in results))
    x_ticks = np.linspace(0, max(len(result) for _, result in results), 5)
    plt.xticks(x_ticks, fontsize=14)

    # 启用网格
    plt.grid(True)

    # # 添加垂直橘色虚线
    # for x in [0, 200, 210, 400]:
    #     plt.axvline(x=x, color='orange', linestyle='--', linewidth=1)

    # # 添加透明橙色填充
    # plt.axvspan(0, 200, color='orange', alpha=0.3)
    # plt.axvspan(210, 400, color='orange', alpha=0.3)

    # 添加图例并设置位置
    plt.legend(loc="upper right", fontsize=13)

    # 保存图像为PDF文件
    plt.savefig('graph.pdf', bbox_inches='tight')

    # 显示绘制的图像
    # plt.show()

# 两个子图的代码-----------------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import numpy as np

# def plot_results(results):
#     # 设置画布尺寸为29:20比例
#     figsize = (29, 20)
#     plt.rcParams['figure.figsize'] = figsize

#     # 定义标记符号和颜色
#     markers = ['s', 'o', 'd', '^', '*', 'p', 'h', 'H', 'v', '<', '>', '8', 'x', 'X', 'D']
#     colors = [
#         'black', 'orangered', 'saddlebrown', 'blue', 'green', 'purple', 'magenta', 
#         'cyan', 'gold', 'darkorange', 'pink', 'lightgreen', 'teal', 'navy', 'maroon'
#     ]

#     # 确保 markers 和 colors 的数量足够
#     num_results = len(results)
#     if num_results > len(markers):
#         markers *= (num_results // len(markers)) + 1
#     if num_results > len(colors):
#         colors *= (num_results // len(colors)) + 1

#     # 绘制每个结果
#     for idx, (name, result) in enumerate(results):
#         Iter_Num = len(result)
#         plt.plot(range(1, Iter_Num + 1), result, 
#                  label=name, linestyle='-', linewidth=5, markersize=20,  # 增加线条宽度和标记大小
#                  markevery=range(Iter_Num // 10, Iter_Num, Iter_Num // 10), 
#                  marker=markers[idx], color=colors[idx])

#     # 设置x/y坐标标签及字体大小自动调整
#     base_fontsize = 24
#     plt.xlabel('Iteration', fontsize=base_fontsize * figsize[0] / 10, fontname='Times New Roman')
#     plt.ylabel(r'$\|x_k - x^*\|$', fontsize=base_fontsize * figsize[0] / 10, fontname='Times New Roman')

#     # 将y轴刻度转换为10的负次方形式（对数刻度）
#     plt.yscale('log')
#     plt.yticks(fontsize=20 * figsize[0] / 10)

#     # 设置x轴刻度范围和间隔，并确保从0开始
#     plt.xlim(0, max(len(result) for _, result in results))
#     x_ticks = np.linspace(0, max(len(result) for _, result in results), 5)
#     plt.xticks(x_ticks, fontsize=20 * figsize[0] / 10)

#     # 启用网格
#     plt.grid(True)

#     # 添加图例并设置位置，增大图例字体和位置
#     plt.legend(loc="upper right", fontsize=20 * figsize[0] / 10)

#     # 保存图像为PDF文件
#     plt.savefig('graph.pdf', bbox_inches='tight')

#     # 显示绘制的图像
#     # plt.show()

# 三个子图的代码-----------------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import numpy as np

# def plot_results(results):
#     # 设置画布尺寸为43:40比例
#     figsize = (43, 40)
#     plt.rcParams['figure.figsize'] = figsize

#     # 定义标记符号和颜色
#     markers = ['s', 'o', 'd', '^', '*', 'p', 'h', 'H', 'v', '<', '>', '8', 'x', 'X', 'D']
#     colors = [
#         'black', 'orangered', 'saddlebrown', 'blue', 'green', 'purple', 'magenta', 
#         'cyan', 'gold', 'darkorange', 'pink', 'lightgreen', 'teal', 'navy', 'maroon'
#     ]

#     # 确保 markers 和 colors 的数量足够
#     num_results = len(results)
#     if num_results > len(markers):
#         markers *= (num_results // len(markers)) + 1
#     if num_results > len(colors):
#         colors *= (num_results // len(colors)) + 1

#     # 绘制每个结果
#     for idx, (name, result) in enumerate(results):
#         Iter_Num = len(result)
#         plt.plot(range(1, Iter_Num + 1), result, 
#                  label=name, linestyle='-', linewidth=10, markersize=43,  # 增加线条宽度和标记大小
#                  markevery=range(Iter_Num // 10, Iter_Num, Iter_Num // 10), 
#                  marker=markers[idx], color=colors[idx])

#     # 设置x/y坐标标签及字体大小自动调整
#     base_fontsize = 30
#     plt.xlabel('Iteration', fontsize=base_fontsize * figsize[0] / 10, fontname='Times New Roman')
#     plt.ylabel(r'$\|x_k - x^*\|$', fontsize=base_fontsize * figsize[0] / 10, fontname='Times New Roman')

#     # 将y轴刻度转换为10的负次方形式（对数刻度）
#     plt.yscale('log')
#     plt.yticks(fontsize=25 * figsize[0] / 10)

#     # 设置x轴刻度范围和间隔，并确保从0开始
#     plt.xlim(0, max(len(result) for _, result in results))
#     x_ticks = np.linspace(0, max(len(result) for _, result in results), 5)
#     plt.xticks(x_ticks, fontsize=25 * figsize[0] / 10)

#     # 启用网格
#     plt.grid(True)

#     # 添加图例并设置位置，增大图例字体和位置
#     plt.legend(loc="upper right", fontsize=30 * figsize[0] / 10)

#     # 保存图像为PDF文件
#     plt.savefig('graph.pdf', bbox_inches='tight')

#     # 显示绘制的图像
#     # plt.show()