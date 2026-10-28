# 脚本入口

`audit_fair_csar.py`：只读统计 `--project` 中 FAIR-CSAR 各版本的样本、标注、跨版本关系和 ZIP 清单，报告写入 `--output`。仅依赖 Python 标准库，完整扫描需数分钟。详细结果见 `records/DATA_AUDIT_2026-10-27.md`。

`verify_annotations.py`：需要 Pillow；全量检查 Full-Pose 九点标注、读取图像尺寸并生成典型样例与边界问题的叠加图，同时模拟旧加载器的几何检查差异。只读数据源，输出目录必须位于数据源以外。详细结论见 `records/ANNOTATION_VERIFICATION_2026-10-27.md`。

`freeze_dataset.py`：核对已审计源标签，对全部 PNG 计算 SHA-256，统一边界规则，隔离类别疑点，按原始大图冻结分组划分，产出派生标签和完整校验索引。

`package_freeze.py`：将上一步生成的哈希与名单打包到 `splits/<version>/`，不把图像或标签本体放入 Git。已有版本目录会拒绝覆盖。

`materialize_dataset.py`：从冻结记录核对源数据，重建 Detect/Pose 数据及绝对路径的 `data.yaml`。两种任务的框和图像一致，默认以硬链接避免重复占用图像磁盘空间。

`test_dataset_policy.py`：关键边界变换、有效点掩码、可复现分组、重复图像隔离与冻结记录篡改检查，可用 `python -m unittest discover -s scripts -p test_dataset_policy.py -v` 运行。

`check_frozen_dataset.py`：对实际生成的两种任务数据逐文件校验标签哈希、目标行、类/框前缀、有效点掩码、链接图像一致性与划分无交集，报告写入指定路径；不替代训练框架加载检查。

`train_e00.py`：Ultralytics 8.3.235 的 E00 入口。`--mode smoke` 使用 64 张 train 和 32 张 val 跑 2 轮，核对实际 loader 图片/目标数、保存模型并重载推理，记录环境及哈希；拒绝覆盖已存在的运行目录。运行方式和正式实验前置条件见 `records/E00_SETUP_2026-10-28.md`。

`test_training_protocol.py`：不依赖 GPU 的训练协议检查，包括拒绝 test/隔离集输入、抽样可复现、固定完整划分计数和验证报告状态/哈希检查。统一运行 `python -m unittest discover -s scripts -p 'test_*.py' -v`。

待实现/执行：远程短训练、断点恢复、共同预训练初始化规则、E01–E03 与独立评估。
