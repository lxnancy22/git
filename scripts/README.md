# 脚本入口

`audit_fair_csar.py`：只读统计 `--project` 中 FAIR-CSAR 各版本的样本、标注、跨版本关系和 ZIP 清单，报告写入 `--output`。仅依赖 Python 标准库，完整扫描需数分钟。详细结果见 `records/DATA_AUDIT_2026-10-27.md`。

`verify_annotations.py`：需要 Pillow；全量检查 Full-Pose 九点标注、读取图像尺寸并生成典型样例与边界问题的叠加图，同时模拟旧加载器的几何检查差异。只读数据源，输出目录必须位于数据源以外。详细结论见 `records/ANNOTATION_VERIFICATION_2026-10-27.md`。

待实现：划分生成、少量数据流程验证、训练与独立评估。正式实验入口应保存完整配置、种子、代码版本和环境信息，并在输出目录已存在时避免覆盖历史结果。
