# FAIR-CSAR 本地数据检查

检查日期：2026-10-27。范围：`D:/SAR/project` 下六个 FAIR-CSAR 目录及四个同名前缀 ZIP。未修改或重新划分任何源数据。

## 目录与规模

| 目录 | trainval 图像/标注对 | test 图像/标注对 | trainval / test 目标数 | 标注 |
| --- | ---: | ---: | ---: | --- |
| FAIR-CSAR-Full-Pose-Ordered | 11959 | 2706 | 69298 / 15987 | 23 列，框 + 九个二维点 |
| FAIR-CSAR-Mini | 1195 | 270 | 6926 / 1450 | 5 列，纯检测 |
| FAIR-CSAR-Mini-Pose | 1195 | 270 | 7237 / 1441 | 23 列 |
| FAIR-CSAR-Mini-Pose-Ordered | 1195 | 270 | 7237 / 1441 | 23 列 |
| FAIR-CSAR-Mini-Pose-XYSort | 1195 | 270 | 7237 / 1441 | 23 列 |
| FAIR-CSAR-Mini-Pose-Aligned | 7188 | 1431 | 7188 / 1431 | 每图一个目标，23 列 |

Full 的真实根目录多嵌套一层：
`D:/SAR/project/FAIR-CSAR-Full-Pose-Ordered/FAIR-CSAR-Full-Pose-Ordered`。

上述数据均检出了类别 ID 0–4。未发现空标注、图像缺标注或标注缺图像，未发现非法数值行或非正框宽高。检查不等于已验证图像内容或标注语义正确。

除 Aligned 为 JPEG 外，其他版本为 PNG。每版本抽查一张训练图，普通版本为 1024×1024 灰度图，Aligned 为 1024×1024 RGB 图。未逐张验证尺寸或图像解码完整性。

## 已确认的版本关系

1. Mini 纯检测版与 Mini-Pose 的样本不同：训练仅重合 115/1195 个文件名，测试仅重合 30/270 个。直接用两者比较检测和 Pose，会同时改变数据。
2. Mini-Pose、Mini-Pose-Ordered、Mini-Pose-XYSort 的 1465 个文件名、目标行顺序、类别和框均一致；逐目标忽略点顺序后，九点坐标集合也全部一致。三版各 split 前三张图的 SHA-256 一致；尚未对全部图像做内容哈希比较。
3. Mini 纯检测版和上述三个 Mini-Pose 版本均可按文件名对应到 Full，重合样本的类别和框一致。Mini-Pose 的点集合也全部可对应到 Full。
4. Full-Ordered 与 Mini-Ordered 的 1465 份重合标注中，1405 份完整数值相同，60 份点顺序不同，点集合保持一致。目录名中的 Ordered 不能当作排序版本标识。
5. Aligned 文件带 `_objN` 后缀，目标数从 Mini-Pose 的 8678 降至 8619，图像编码、坐标也发生变化。只能确认是独立派生版本；缺少转换脚本，暂不能确定旋转、筛选等全部处理规则。

## 标注边界问题

| 版本 | trainval 框越界目标 | test 框越界目标 | trainval 点越界目标 | test 点越界目标 |
| --- | ---: | ---: | ---: | ---: |
| Full-Ordered | 45 | 6 | 26 | 3 |
| Mini | 2 | 0 | — | — |
| Mini-Pose / Ordered / XYSort（各版） | 5 | 1 | 4 | 0 |
| Aligned | 2 | 0 | 470 | 107 |

这里“框越界”表示存储的 cx、cy、w、h 至少一项不在 [0,1]，不是把 cx±w/2 超边界也判为错误；“点越界”指至少一个点坐标不在 [0,1]。两个问题可发生于同一目标，不能直接相加当作独立目标总数。

Full 点越界涉及 trainval 25 份、test 3 份标注文件；框数值越界分别涉及 45 份、6 份文件。样例中有负点坐标、点坐标大于 1 和宽度 1.019531，需追溯切片截断及转换规则，不应直接静默裁剪。

## 类别与来源风险

- 旧 `FAIR-CSAR-Mini.zip` 中 `fair_sar.yaml` 声明 nc=4，只列出 Aircraft、Ship、Oil tank、Bridge；但当前 Mini 标签实际存在类别 4。旧配置不可直接复用。
- 旧工程 `sar_pose.yaml` 声明五类，第五类为 Tower Crane。类别名称仍需与原始官方标签或可信转换脚本核对，当前数值检查不能证明语义映射。
- 这六个目录及四个 ZIP 中仅发现派生图像、YOLO TXT、缓存和一份旧 YAML，未发现原始 SLC 复数文件、官方 XML/KeyPoints 或成像元数据。不能将 Full 目录名解释为完整官方原始数据。
- 未在本次搜索的 Python 脚本中找到可确认这些 FAIR-CSAR 版本生成与排序过程的实现；找到的 `my_converter.py` 处理 MSAR，不是 FAIR-CSAR。

## 划分检查与边界

按文件名去掉 `_objN` 及最后两个整数切片偏移推断原始图像前缀：Full trainval 为 105 组，test 为 26 组，组间无相同前缀；各 Mini 版本为 102 / 25 组，Aligned 归并回原图后同样为 102 / 25 组。

各版本 trainval/test 之间无相同标注文件名。上述检查未覆盖完整图像内容去重、不同日期同地理区域重叠或官方场景元数据核验，不能宣称完全无泄漏。所有版本目前仅有 trainval/test，没有独立 val。

## 建议下一步

1. 暂以 Full-Ordered 作为可追溯的数据候选，保留原目录只读使用；先核对类别映射、越界标签和排序来源。
2. 从同一份确认后的九点数据派生纯检测标签，保留相同图像、框与划分，避免使用不同抽样的旧 Mini 对照。
3. 从 trainval 按原图/场景分组建立固定 train/val，保留现有 test 作最终测试；先比较类别与目标数量分布，再冻结名单和哈希。
4. 从该统一来源按固定名单产生调试子集；Aligned 暂不加入普通 DCN 与点引导 DCN 的核心对比。
5. 如需研究物理散射中心或复数域特征，另行找回原始 SLC 与官方标注，不能从 PNG 恢复相位。

## 记录与复查

- `fair-csar-inventory.json`：全部 TXT 数值统计、跨版本比较、样例图像哈希和 ZIP 目录清单。
- `fair-csar-scene-inventory.json`：按文件名推断的场景数量与交集。
- `fair-csar-boundary-examples.json`：Full 越界行的文件名、行号和数值。

主清单可通过标准库脚本重新生成：

```powershell
python scripts/audit_fair_csar.py --project D:/SAR/project --output records/fair-csar-inventory.json
```

该脚本只读源数据，唯一写入目标是指定的 JSON。完整扫描需数分钟。场景和越界明细为本次补充检查结果，尚未集成至主脚本。

补充回查：扩大到旧项目外层后发现 `project/yolov10-main/PCA_wash_dataset.py` 和 `align_dataset.py`。此前脚本搜索范围不完整，详见后续 `DATA_FREEZE_2026-10-27.md`。数据整理与划分现已作为独立 v1 冻结，本文保留初次检查时的原始统计。
