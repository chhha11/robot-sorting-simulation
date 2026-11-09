# 视觉极端场景对照演示

运行 `python main.py --visual-test`，打开左右分屏：左侧调用项目原有的 OpenCV HSV 颜色阈值与轮廓检测，右侧默认显示**模拟理想框**。屏幕显示各侧 `Found X objects`，并轮播以下五种场景。

| 场景 | 内容 |
| --- | --- |
| `clean` | 分离摆放的红蓝方块，作为正常条件参考 |
| `crowded` | 同色紧邻、密集摆放与上下堆叠，形成图像交叠 |
| `occluded` | 密集方块上方增加真实灰色遮挡条 |
| `dim` | 密集场景叠加低曝光与传感器噪声 |
| `glare` | 密集场景叠加移动的白色光晕，造成颜色淡化 |

两侧显示同一帧、同一份退化后的 RGB。场景使用固定物块与曝光扫描，避免物体运动改变对照条件；此模式使用独立仿真，不启动机械臂或调度器，也不修改正常分拣场景。

**先区分两种右侧模式。**

- `--visual-yolo mock`（默认）：直接把仿真可见物体标注包装成 YOLO 的 `N×6` 框格式。画面持续标注 `SIMULATED / IDEAL BOXES - NOT NEURAL INFERENCE`，模型准确率显示 `N/A`，JSON 中 `yolo_metrics` 与汇总 `yolo` 均为 `null`。它适合展示检测接口和理想结果，不能作为“YOLO 优于颜色阈值”的实验证据。
- `--visual-yolo real`：使用 Ultralytics 对退化后的 RGB 做真实神经网络推理。模型不接收仿真坐标、分割图、深度或真值。需提供已训练的红蓝方块权重；通用 COCO `yolov8n.pt` 没有这些类别，程序会拒绝将其作为本实验模型。真实模型表现由权重与场景决定，不预设其一定胜出。

**启动、录屏与导出。**

```powershell
# 默认演示：循环播放，按 Q 退出
python main.py --visual-test

# 录制一个完整的 25 秒场景周期，保存分屏 MP4、截图与 JSON
python main.py --visual-test --visual-duration 25 --visual-record outputs/visual_test/demo.mp4 --save-dir outputs/visual_test

# 无窗口生成同样的录屏与结果
python visual_test.py --headless --visual-duration 25 --visual-record outputs/visual_test/headless.mp4 --save-dir outputs/visual_test

# 使用自己训练的红蓝方块模型，进行真实检测对照
python main.py --visual-test --visual-yolo real --model runs/detect/train/weights/best.pt --visual-duration 25 --visual-record outputs/visual_real/demo.mp4 --save-dir outputs/visual_real

# 固定低照度场景
python main.py --visual-test --visual-scenario dim
```

默认每个场景停留 5 秒，录制帧率为 15 FPS；可用 `--visual-period` 与 `--visual-fps` 调整。`--visual-duration` 指录制时间轴长度，推理速度较慢时实际运行耗时可能更长。GUI 不指定时长则循环；无窗口不指定时长则运行 30 秒。`--seed` 固定物块类别排列与噪声，便于复现。

真实模型类别名可为 `Box_A` / `Box_B`、`red` / `blue` 或 `red_box` / `blue_box` 等受支持别名，且必须同时包含红、蓝两类。替换命令中的权重路径为实际文件；此工具不自动训练模型。

| 快捷键 | 功能 |
| --- | --- |
| `1` ～ `5` | 依次切换正常、密集、遮挡、低照度、眩光 |
| `A` | 恢复自动轮播 |
| `Space` | 暂停 / 继续 |
| `S` | 保存当前分屏截图 |
| `Q` / `Esc` | 结束并保存结果 |

输出目录包含各场景首次出现的截图、`last_frame.png` 与 `metrics.json`；指定 `--visual-record` 时另写入所选 MP4 路径。JSON 记录模式、种子、每帧统计和分场景汇总，真实模式还记录权重路径与 SHA-256。

**统计口径。**

默认采用逐帧、同类别、IoU ≥ 0.5 的一对一匹配：按检测置信度从高到低匹配最佳未占用真值框，重复框会计入 FP；`P = TP / (TP + FP)`，`R = TP / (TP + FN)`，空分母记为 0。可通过 `--visual-iou` 改变阈值。分场景汇总先累加 TP / FP / FN，再计算 P / R，避免将空帧当成满分。

真值框包围当前分割图中**实际可见的物体像素**。物体至少有 20 个可见像素，且达到同视角孤立轮廓面积的 15%，才纳入评分；完全遮挡的物体不计漏检。仿真标注仅用于评价，以及明确标注的 mock 理想框。

此口径有两点限制：真实 YOLO 若按完整物体框（amodal）训练，在遮挡时可能预测出完整范围，与可见框计算 IoU 会受到额外惩罚；低于可见性门槛的物体被移出真值集，但若检测器仍报出它，其框仍按普通未匹配检测计为 FP，并没有单独的 ignore 区域。跨模型比较前应统一训练标注与评价规则。

这里的 P / R 是**逐帧物体检测指标**，不是 mAP、唯一物体计数或机械臂分拣成功率。录屏或汇报时请保留模式说明；正式展示 YOLO 识别成功率必须使用真实模式与合适的训练权重。
