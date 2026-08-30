# MPE环境参数配置文档
# Hybrid MAPPO-GCBF+ for Multi-Agent Particle Environment (MPE)
# 项目实际使用的完整参数配置

本文档记录了Hybrid MAPPO-GCBF+项目中实际使用的所有MPE环境参数配置。

## 🌍 环境基础参数

### 世界物理属性
```yaml
# 空间设置
area_size: 2.0                    # 世界大小 (2×2米的正方形)
world_bounds: [-1.0, 1.0]         # 世界边界坐标 [min, max]
dim_p: 2                          # 位置维度 (x, y)
dim_c: 2                          # 通信维度

# 时间设置
dt: 0.1                           # 物理仿真时间步长 (秒)
max_episode_length: 25            # 最大回合长度 (时间步)
total_episode_time: 2.5           # 总回合时间 (秒) = 25 × 0.1

# 动作空间
discrete_action: false            # 使用连续动作空间
action_dim: 2                     # 每个智能体的动作维度 [force_x, force_y]
action_range: [-5.0, 5.0]         # 环境动作范围 (牛顿)
policy_output_range: [-1.0, 1.0]  # 策略输出范围
action_scaling_factor: 5.0        # 动作缩放因子
```

### 物理仿真参数
```yaml
# 动力学模型
integrator: "euler"               # 欧拉积分器
damping: 0.25                     # 阻尼系数
contact_force: 100.0              # 接触力系数
contact_margin: 0.001             # 接触边距

# 控制动力学 (控制仿射系统)
# dx/dt = f(x) + g(x)u
# f(x) = [vx, vy, 0, 0]          # 漂移项
# g(x) = [[0,0], [0,0], [1,0], [0,1]]  # 控制矩阵
```

## 🤖 智能体参数配置

### 基本物理属性
```yaml
# 几何属性
agent_size: 0.05                  # 智能体半径 (米)
agent_diameter: 0.10              # 智能体直径 (米)

# 质量属性
mass: 1.0                         # 质量 (kg)
initial_mass: 1.0                 # 初始质量 (kg)
density: 25.0                     # 材料密度 (kg/m³)

# 运动能力
u_range: 1.0                      # 控制输入范围 [-1, 1]
max_speed: null                   # 最大速度 (无限制)
max_acceleration: 5.0             # 最大加速度 (m/s²) = max_force/mass
accel_coefficient: 5.0            # 加速度系数
```

### 智能体能力设置
```yaml
# 基本能力
movable: true                     # 可移动
collide: true                     # 可碰撞
adversary: false                  # 非对抗性
dummy: false                      # 非虚拟智能体

# 感知能力
silent: false                     # 可通信
blind: false                      # 可观测
observation_range: 2.0            # 观测范围 (覆盖整个世界)

# 噪声设置
u_noise: null                     # 动作噪声 (无)
c_noise: null                     # 通信噪声 (无)
```

## 🏷️ 地标参数配置

### 地标物理属性
```yaml
# 几何属性
landmark_size: 0.05               # 地标半径 (米)
landmark_diameter: 0.10           # 地标直径 (米)

# 物理属性
landmark_mass: 1.0                # 地标质量 (kg)
landmark_density: 25.0            # 地标密度 (kg/m³)
landmark_movable: false           # 地标不可移动
landmark_collide: false           # 地标不参与碰撞
landmark_boundary: false          # 地标非边界对象
```

## 🛡️ 安全约束参数

### 碰撞避免约束
```yaml
# Simple Spread场景
simple_spread:
  min_distance: 0.18              # 智能体间最小安全距离 (米)
  safety_margin: 0.04             # 额外安全边距 (米)
  total_safe_distance: 0.22       # 总安全距离 = min_distance + safety_margin
  collision_threshold: 0.10       # 碰撞检测阈值 (agent_size × 2)
  
# Simple Navigation场景 (稍微严格)
simple_navigation:
  min_distance: 0.16              # 智能体间最小安全距离 (米)
  safety_margin: 0.04             # 额外安全边距 (米)
  total_safe_distance: 0.20       # 总安全距离
  collision_threshold: 0.10       # 碰撞检测阈值

# 约束类型和参数
constraint_type: "soft_penalty"    # 软约束类型
penalty_weight: 8.0               # 违反惩罚权重
barrier_function: "exponential"    # 障碍函数类型
```

### 边界约束
```yaml
# Simple Spread场景
simple_spread:
  boundary_buffer: 0.08           # 边界缓冲区宽度 (米)
  safe_area_bounds: [-0.92, 0.92] # 安全区域边界
  
# Simple Navigation场景 (稍微严格)
simple_navigation:
  boundary_buffer: 0.06           # 边界缓冲区宽度 (米)
  safe_area_bounds: [-0.94, 0.94] # 安全区域边界

# 约束参数
constraint_type: "exponential"     # 渐进式约束
strength_factor: 1.5              # 约束强度因子
violation_penalty: 5.0            # 违反惩罚权重
```

## 🎯 场景特定参数

### Simple Spread场景
```yaml
scenario_name: "simple_spread"
num_agents: 3                     # 智能体数量
num_landmarks: 3                  # 地标数量
max_episode_length: 25            # 最大回合长度

# 任务参数
task_type: "coverage"             # 任务类型：覆盖
success_criteria: "all_landmarks_covered"  # 成功标准
coverage_threshold: 0.1           # 覆盖距离阈值 (米)
success_threshold: 0.1            # 成功判定阈值

# 奖励设计
reward_components:
  coverage_reward: -1.0           # 到最近地标距离的负值
  collision_penalty: -1.0         # 每次碰撞的惩罚
  step_penalty: 0.0               # 时间步惩罚
  boundary_penalty: -0.1          # 边界违反惩罚

# 初始化设置
agent_init_pos: "random"          # 智能体初始位置：随机
landmark_init_pos: "random"       # 地标初始位置：随机
init_pos_range: [-1.0, 1.0]       # 初始位置范围
```

### Simple Navigation场景
```yaml
scenario_name: "simple_navigation"
num_agents: 3                     # 智能体数量
num_landmarks: 3                  # 目标点数量 (用作导航目标)
max_episode_length: 25            # 最大回合长度

# 任务参数
task_type: "navigation"           # 任务类型：导航
success_criteria: "all_agents_arrived"  # 成功标准
arrival_threshold: 0.1            # 到达距离阈值 (米)
success_threshold: 0.1            # 成功判定阈值

# 奖励设计
reward_components:
  distance_reward: -1.0           # 到目标距离的负值
  collision_penalty: -1.0         # 每次碰撞的惩罚
  boundary_penalty: -0.1          # 边界违反惩罚
  step_penalty: 0.0               # 时间步惩罚

# 目标分配
target_assignment: "random"       # 目标随机分配
target_init_pos: "random"         # 目标位置随机
```

## 🔗 图结构参数

### GraphsTuple结构定义
```yaml
# 节点相关
node_dim: 6                       # 节点特征维度
node_features:
  simple_spread: "[x, y, vx, vy, closest_landmark_x, closest_landmark_y]"
  simple_navigation: "[x, y, vx, vy, goal_x, goal_y]"
node_type: [0, 0, 0]              # 节点类型 (全部为智能体)

# 边相关
edge_dim: 4                       # 边特征维度
edge_features: "[distance, relative_x, relative_y, relative_speed]"
connectivity_pattern: "fully_connected"  # 全连接图
n_edges: 6                        # 边数量 (3个智能体 × 2方向)

# 状态相关
state_dim: 4                      # 状态维度 [x, y, vx, vy]
env_state_dim: 3                  # 环境状态维度
```

### 图连接模式
```yaml
# 3智能体全连接有向图
graph_topology:
  nodes: [0, 1, 2]                # 节点索引
  edges:
    - {sender: 0, receiver: 1}    # 智能体0 → 智能体1
    - {sender: 0, receiver: 2}    # 智能体0 → 智能体2
    - {sender: 1, receiver: 0}    # 智能体1 → 智能体0
    - {sender: 1, receiver: 2}    # 智能体1 → 智能体2
    - {sender: 2, receiver: 0}    # 智能体2 → 智能体0
    - {sender: 2, receiver: 1}    # 智能体2 → 智能体1
```

## 🧠 观测空间配置

### 个体观测空间
```yaml
# Simple Spread
simple_spread:
  agent_obs_dim: 16               # 4 + 2×(3-1) + 2×3 = 4+4+6
  obs_components:
    - self_velocity: [2]          # 自身速度 [vx, vy]
    - self_position: [2]          # 自身位置 [x, y]
    - landmark_rel_pos: [6]       # 地标相对位置 [2×3]
    - other_agent_rel_pos: [4]    # 其他智能体相对位置 [2×2]

# Simple Navigation
simple_navigation:
  agent_obs_dim: 10               # 6 + 2×(3-1) = 6+4
  obs_components:
    - self_velocity: [2]          # 自身速度 [vx, vy]
    - self_position: [2]          # 自身位置 [x, y]
    - goal_rel_pos: [2]           # 目标相对位置 [x, y]
    - other_agent_rel_pos: [4]    # 其他智能体相对位置 [2×2]
```

### 全局状态空间
```yaml
# Simple Spread
simple_spread:
  global_state_dim: 18            # 4×3 + 2×3 = 12+6
  state_components:
    - all_agent_states: [12]      # 所有智能体状态 [4×3]
    - all_landmark_pos: [6]       # 所有地标位置 [2×3]

# Simple Navigation  
simple_navigation:
  global_state_dim: 18            # 4×3 + 2×3 = 12+6
  state_components:
    - all_agent_states: [12]      # 所有智能体状态 [4×3]
    - all_goal_pos: [6]           # 所有目标位置 [2×3]
```

## 🏋️ 训练参数配置

### GCBF+特定参数
```yaml
# 网络架构
gnn_layers: 1                     # GNN层数
batch_size: 256                   # 批量大小
buffer_size: 512                  # 经验回放缓冲区大小
horizon: 32                       # 安全预测时域

# 学习率
lr_actor: 1.0e-5                  # Actor学习率
lr_cbf: 1.0e-5                    # CBF学习率

# CBF参数
alpha: 1.2                        # CBF类别-K函数参数
eps: 0.015                        # 约束松弛参数
inner_epoch: 8                    # 内部训练轮数
max_grad_norm: 2.0                # 最大梯度范数

# 损失函数权重
loss_action_coef: 0.001           # 动作损失系数
loss_unsafe_coef: 1.0             # 不安全损失系数
loss_safe_coef: 1.0               # 安全损失系数
loss_h_dot_coef: 0.2              # CBF时间导数损失系数
```

### MAPPO特定参数
```yaml
# 网络架构
hidden_size: 64                   # 隐藏层大小
layer_N: 1                        # 网络层数
use_orthogonal: true              # 使用正交初始化
activation_id: 1                  # 激活函数 (ReLU)

# 训练参数
learning_rate: 3.0e-4             # 学习率
critic_lr: 3.0e-4                 # Critic学习率
ppo_epochs: 10                    # PPO训练轮数
clip_param: 0.2                   # PPO裁剪参数
entropy_coef: 0.01                # 熵系数
value_loss_coef: 1.0              # 价值损失系数
max_grad_norm: 10.0               # 最大梯度范数

# 高级选项
use_recurrent_policy: false       # 不使用循环策略
use_naive_recurrent_policy: false # 不使用朴素循环策略
use_max_grad_norm: true           # 使用梯度裁剪
use_clipped_value_loss: true      # 使用裁剪价值损失
use_huber_loss: true              # 使用Huber损失
use_valuenorm: true               # 使用价值归一化
```

### 混合训练参数
```yaml
# 权重分配
u_ref_weight:                     # 参考策略跟随权重
  simple_spread: 0.7              # 平衡任务和安全
  simple_navigation: 0.8          # 更注重任务完成

safety_weight:                    # 安全约束权重
  simple_spread: 0.3              # 与u_ref_weight互补
  simple_navigation: 0.2          # 与u_ref_weight互补

# 更新频率
mappo_update_freq: 1              # MAPPO更新频率
gcbf_update_freq: 1               # GCBF+更新频率

# 训练阶段
total_steps:                      # 总训练步数
  simple_spread: 2000
  simple_navigation: 2500

mappo_pretrain_steps:             # MAPPO预训练步数
  simple_spread: 500
  simple_navigation: 600

# 并行环境
n_env_train: 16                   # 训练环境数量
n_env_test: 8                     # 测试环境数量
```

## 🔧 系统配置参数

### 运行时配置
```yaml
# 设备设置
device: "auto"                    # 自动选择设备 (cpu/cuda)
seed: 42                          # 随机种子

# 日志和保存
log_interval: 10                  # 日志记录间隔
eval_interval: 50                 # 评估间隔
save_interval: 100                # 模型保存间隔

# 评估设置
eval_episodes: 10                 # 评估回合数
eval_deterministic: true         # 评估时使用确定性策略
```

### 数据类型配置
```yaml
# JAX数据类型
jax_dtype: "float32"              # JAX浮点数类型
jax_int_dtype: "int32"            # JAX整数类型

# PyTorch数据类型
torch_dtype: "float32"            # PyTorch浮点数类型
torch_device: "cpu"               # PyTorch设备 (自动检测GPU)
```

## 📊 性能评估参数

### 成功判定标准
```yaml
# Simple Spread
simple_spread:
  success_criteria:
    - all_landmarks_covered: true         # 所有地标被覆盖
    - no_collisions: true                 # 无碰撞发生
    - within_boundaries: true             # 在边界内
    - max_episode_steps: 25               # 在规定步数内完成

# Simple Navigation
simple_navigation:
  success_criteria:
    - all_agents_arrived: true           # 所有智能体到达目标
    - no_collisions: true                 # 无碰撞发生
    - within_boundaries: true             # 在边界内
    - max_episode_steps: 25               # 在规定步数内完成
```

### 评估指标
```yaml
# 任务性能指标
task_metrics:
  - success_rate                  # 成功率
  - coverage_rate                 # 覆盖率 (Simple Spread)
  - arrival_rate                  # 到达率 (Simple Navigation)
  - average_episode_length        # 平均回合长度
  - task_completion_time          # 任务完成时间

# 安全性能指标
safety_metrics:
  - collision_rate                # 碰撞率
  - boundary_violation_rate       # 边界违反率
  - safety_constraint_satisfaction # 安全约束满足率
  - min_distance_violations       # 最小距离违反次数
  - safety_score                  # 综合安全得分

# 效率指标
efficiency_metrics:
  - computational_time            # 计算时间
  - action_smoothness            # 动作平滑度
  - energy_consumption           # 能量消耗 (动作模长积分)
  - path_efficiency              # 路径效率
```

---

## 📝 参数使用说明

### 参数优先级
1. 命令行参数 > 配置文件参数 > 默认参数
2. 场景特定参数 > 通用参数
3. 安全约束参数不可被覆盖 (硬编码保护)

### 参数验证
- 所有距离参数必须为正数
- 安全距离必须大于智能体直径
- 权重参数之和应为1.0
- 训练步数必须为正整数

### 参数调试建议
- 从默认参数开始，逐步调整
- 安全约束过严可能导致任务无法完成
- 学习率过大可能导致训练不稳定
- 批量大小影响训练效率和稳定性

---

*最后更新：项目创建时*
*参数来源：实际代码实现*
