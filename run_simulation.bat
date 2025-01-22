@echo off
REM 激活虚拟环境（如果有）
REM call venv\Scripts\activate

REM 运行仿真
python simulation.py --config config\default_config.yaml
