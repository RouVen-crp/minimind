# Configs

每次运行保存一份不可变配置，建议命名为 `YYYYMMDD-stage-variant.json`。至少记录：数据路径及 SHA-256、起始权重、seed、epochs、batch size、梯度累积、序列长度、学习率、dtype、保存间隔、设备和输出目录。

训练脚本目前使用命令行参数，因此配置文件同时保存对应的完整命令字符串。
