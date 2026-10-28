# FAIR-CSAR v2：重复框标注统一

v2 已在本地从不可变的 `faircsar_local_v1` 派生，保留同图同类同框两种九点标注中先出现的一行，同时在 Detect/Pose 删除另一行。六个精确位置和源标签哈希见 `FAIRCSAR_DUPLICATE_RESOLUTION_2026-10-28.json`。六例均为 class 2 油罐；本规则解决记录一致性，不代表先出现的九点一定更准确。

v1 源图片及其硬链接未改动。v2 场景归属、图像名单、边界规则、隔离样本均沿用 v1。训练集 9582 图、55900 目标；验证集 2377 图、13392 目标；测试集 2701 图、15982 目标。两种任务标签逐文件相同的类别和框前缀。冻结清单位于 `splits/faircsar_local_v2`；manifest SHA-256 为 `75e7f53d9d5f7b9199bd8f60dc7f952205daff50b86654747c818183776da637`。

本地 `scripts/check_frozen_dataset.py` 对 14660 对 Detect/Pose 文件验证通过，报告在忽略 Git 的 `outputs/v2-validation.json`。这验证派生数据的标签哈希、文件配对、类别数、掩码和划分关系。服务器上仍需重建并运行相同验证；远程框架加载数也需再次检查。

服务器执行：

```bash
cd /root/sar-scattering-detection
git pull --ff-only
python -u scripts/derive_faircsar_v2.py --source /root/datasets/faircsar_local_v1 --output /root/datasets/faircsar_local_v2 --freeze-output splits/faircsar_local_v2
python scripts/check_frozen_dataset.py --freeze splits/faircsar_local_v2 --data /root/datasets/faircsar_local_v2 --report outputs/server-data-validation-v2.json
```

`--output` 必须是尚不存在的目录；失败时保留日志和部分输出供排查，不对 v1 就地改写。v2 图片与 v1 共用硬链接；新增数据占用主要来自标签及元数据，不复制两份图片。

远程报告 `PASS` 后使用 `--data-root /root/datasets/faircsar_local_v2 --validation-report outputs/server-data-validation-v2.json` 启动完整 E00，训练入口要求 `configs/e00_protocol_v2.json` 和 v2 manifest 哈希。失败的 `E00_full_s0_001` 保留原日志/记录；重试使用新的 `E00_full_s0_002` 名称。不要用 v1 报告启动 v2。
