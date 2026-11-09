# 双 UR5 视觉分拣仿真

基于 Python、PyBullet 和 OpenCV 的双机械臂传送带分拣仿真。系统通过仿真 RGB-D 相机定位与跟踪红、蓝物块，预测移动目标的抓取位置，向两台 UR5 分配任务，并将物块送入各自的分类放置区。

## 功能

- 连续进料、移动目标预测、双臂独立派单与任务接力
- 共享抓取区域的通行控制、避碰、主动让行与恢复
- 抓取失败、目标丢失、不可达和碰撞等故障注入
- 静态与动态场景的无界面回归，以及视觉场景对照

## 运行

安装 Python 3 和依赖：

```powershell
python -m pip install -r requirements.txt
```

启动交互式传送带仿真：

```powershell
python main.py
```

运行有限数量物块的无界面验证：

```powershell
python main.py --headless --objects 6 --require-both-arms --require-parallel-motion
```

其他入口：

```powershell
python main.py --static --headless
python main.py --visual-test
python -m unittest discover -p "test_*.py"
```

## 模式说明

默认 `--detector yolo` 使用颜色检测生成 YOLO 格式的模拟检测框，并非神经网络推理。`--detector color` 使用 OpenCV 颜色检测；`--detector yolo-real` 需要自行提供包含红、蓝物块类别的训练权重。仓库自带的通用 `yolov8n.pt` 不具备这两个类别。视觉对照模式的具体口径见 [VISUAL_TEST.md](VISUAL_TEST.md)。

`--robot sim` 使用 PyBullet。`--robot real` 当前只有接口框架，尚未接入真实机械臂，详见 [ROBOT_BACKENDS.md](ROBOT_BACKENDS.md)。

UR5 模型的来源及许可见 [assets/ur5/README.md](assets/ur5/README.md) 和 [assets/ur5/LICENSE](assets/ur5/LICENSE)。
