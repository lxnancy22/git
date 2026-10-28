# E00 第一版完整训练协议

冻结日期：2026-10-28（Asia/Shanghai）。状态：完整训练启动入口已准备，尚未收到正式训练结果。

## 前置检查结果

用户提供的服务器终端显示 `E00_smoke_s0_001` 与 `E00_resume_s0_001` 都输出 `PASS`；后者在新进程中恢复 epoch、optimizer、EMA，并完成 3 轮。原始完整日志和运行记录保存在服务器，尚未同步为本地备份。

固定初始化文件来自官方 release `ultralytics/assets v8.3.0`：

- URL：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt>
- 文件大小：22588772 bytes。
- 用户提供的服务器 SHA-256：`1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`。
- 本项目冻结这份下载副本；该 SHA 是本地计算值，不是上游发布的 digest。

## 固定内容

机器可读锁：`configs/e00_protocol_v1.json`，参数：`configs/e00_yolov8s.json`。启动 full 前会核对配置的规范 JSON 哈希、冻结 manifest、初始化文件大小和 SHA-256。配置改变需建立新协议，并重新做可比的基线。

| 项目 | 固定值 |
| --- | --- |
| 模型/框架 | 普通 YOLOv8s Detect / Ultralytics 8.3.235 |
| 初始化 | 锁定的 COCO 预训练权重；训练数据 5 类，匹配层迁入，新/形状不匹配层依种子初始化 |
| 数据 | faircsar_local_v1：train 9582 图/55906 目标，val 2377 图/13392 目标 |
| 输入 | 灰度 PNG 经框架按 3 通道加载；1024×1024，统一框架归一化与 letterbox |
| 轮数 | 100，关闭提前停止 |
| 微批量 / nbs | 8 / 64，采用上游梯度累积规则，warmup 期间累积步数动态变化 |
| 优化器 | SGD，lr0=0.01，momentum=0.937，weight_decay=0.0005 |
| 调度 | 3 轮 warmup，线性衰减，lrf=0.01 |
| 精度/加载 | FP32、workers=4、cache=False |
| 增强 | 平移 0.1、缩放 0.5；禁用旋转、翻转、HSV、混图、额外模糊和 CLAHE |
| 损失 | box=7.5，cls=0.5，dfl=1.5 |
| 验证 | conf=0.001、NMS IoU=0.7、max_det=300；每轮 val，不读取 test |
| 选权重 | 最高 val Box mAP50:95，保持上游相同分数时的保存行为 |
| 种子 | 本次先运行 0；正式核心对比使用共同的 0、1、2 |

本协议用于第一轮可比基线，不声称为最优超参数。E01–E03 实现时共同层须复用同一初始化文件并核对初始参数一致，新增分支初始化需登记；训练设置和 Box 选权重规则保持一致。

## 启动与查看

```bash
conda activate yolov10
cd /root/sar-scattering-detection
git pull --ff-only
mkdir -p outputs
nohup python -u scripts/train_e00.py --mode full --name E00_full_s0_001 --seed 0 --weights weights/pretrained/v8.3.0/yolov8s.pt > outputs/E00_full_s0_001.log 2>&1 < /dev/null &
tail -n 60 -f outputs/E00_full_s0_001.log
```

`nohup` 使进程能在 SSH 连接断开后继续，不能使关机或释放实例后的训练继续。Ctrl+C 只结束前台 tail；不要重复启动同名 run。入口拒绝覆盖已有 run，但日志重定向会截断同名 `.log`，因此重试前应保留原日志并使用新的 run 编号与日志文件名。

第一轮结束后先检查加载数、损失、val 指标、耗时及显存。短训练的运行时间和预测测速不用于估计全量 100 轮时间。若失败，保留日志和 run-record，不把有错误的结果当作基线。

结果在 `runs/E00_full_s0_001/`，包括请求参数、框架参数、依赖和源码哈希、图片名单、实际 loader 数量、CSV、best/last 及最终哈希。结束后还需同步完整日志和权重到本地/持久存储；Git 不保存这些大文件。
