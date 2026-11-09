# 机械臂控制后端

默认运行仿真，不需要修改已有启动命令：

```powershell
python main.py --robot sim
python main.py --robot sim --static --headless
```

`main.py` 通过 `create_robot_arm()` 工厂选择实现。调度器接收
`BaseRobotArm` 接口，抓取位置和放置位置使用统一的世界坐标（米）。

- `BaseRobotArm`：任务提交、非阻塞更新、空闲判断、错误读取及调度所需的状态接口。
- `SimRobotArm`：原有 PyBullet 控制器，包含 IK、电机、夹爪、碰撞避让和状态机。
- `RealRobotArm`：真实驱动接入框架；接入点用于后续 Socket 或 ROS 通信。

`robot_arm.py` 是不依赖 PyBullet 的公共入口；原仿真控制代码迁入
`sim_robot_arm.py`，仅在请求仿真实现时加载。旧的
`from robot_arm import RobotArm` 仍可用，`RobotArm` 是 `SimRobotArm` 的兼容别名。
`SharedWorkspaceGate` 也保留原导入方式，属于仿真后端的碰撞与交通实现。

除基础任务方法外，接口还包含 TCP 查询、取消、抓取耗时估计和目标跟踪更新。
故障接力通过 `ArmHandoffStatus` 获取是否持物、是否可接力及恢复状态，
调度器不读取 Bullet 物体句柄或控制器的私有状态；新驱动未确认可接力时默认禁止接力。

真机路径可用以下命令检查：

```powershell
python main.py --robot real
```

当前真实驱动尚未实现，程序会在创建仿真环境之前报告驱动未就绪并以非零状态退出。
它不会连接真实设备、运行仿真替代真机或虚报抓取成功。选择该路径和导入
`robot_arm` 接口不需要安装 PyBullet；仿真依赖只在选择仿真时加载。

后续接入时，需要在真实驱动中实现连接及就绪检查、读取 TCP、提交轨迹和夹爪命令、
读取执行结果及故障。`update()` 应每次处理一次通信或状态轮询后立即返回，
不能阻塞等待完整运动结束，这样双臂独立更新和故障接力逻辑才能继续工作。
还需提供真实相机、手眼标定、工位坐标与应用主循环；当前仿真主循环中的虚拟夹持和
Bullet 场景验证只用于仿真，不属于硬件接口。

视觉演示模式保持独立：`python main.py --visual-test` 只运行识别对比场景。
