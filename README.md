distributed_optimization_framework/
│
├── README.md
├── run_simulation.sh       # 启动运行脚本
├── config/                 # 配置文件（图结构和算法类型都在这里配置）
│   └── default_config.yaml
├── algorithms/             # 算法实现代码
│   ├── DAGT_Algorithm.py
|   ├── MDGT_Algorithm.py
|   ├── DACS_Algorithm.py
|   ├── DACS_HB_Algorithm.py
├── network_topologies/     # 网络拓扑生成代码
│   ├── Fixed_Matrix.py
│   └── Time_varying_Matrix.py
├── visualizations/         # 画图代码
│   └── plot_results.py
└── simulation.py           # 框架的主函数

作者：Qixing Zhou
有bug请联系：qixingzhou1125@outlook.com