# GAT-MF + GCBF+ 多智能体安全控制项目

本项目实现了基于图注意力网络的多智能体强化学习（GAT-MF）与图控制屏障函数（GCBF+）的两阶段训练管道，旨在在保证到达率的基础上最大化安全率。

## 🎯 项目目标

1. **第一阶段（GAT-MF）**: 训练基础策略达到80%左右的到达率
2. **第二阶段（GCBF+）**: 使用GAT-MF作为参考策略，在保证到达率的基础上提高安全率

## 🏗️ 项目结构

```
gatmf_gcbf_new-main/
├── 📁 grid_gatmf_gcbf+/           # 核心算法模块
│   ├── 📁 gat_mf/                 # GAT-MF算法实现
│   │   ├── gat_mf_train_eval_fixed.py  # 主训练器
│   │   ├── grid_model.py              # 3×3网格环境
│   │   └── grid_networks.py           # 神经网络架构
│   ├── 📁 gcbf_plus/              # GCBF+算法实现
│   │   ├── 📁 algo/               # 算法核心
│   │   ├── 📁 env/                # 环境定义
│   │   ├── 📁 trainer/            # 训练器
│   │   └── 📁 utils/              # 工具函数
│   ├── 📁 convert/                # PyTorch→JAX转换工具
│   │   └── pytorch_to_jax_converter.py
│   └── 📁 jaxproxqp-master/       # QP求解器
├── 🚀 train_full_pipeline.py      # 完整训练管道
├── 🎯 train_gcbf_plus.py          # GCBF+单独训练
├── 📊 可视化工具/
│   ├── visualize_web.py           # Web可视化（推薦）
│   ├── visualize_text.py          # 終端可视化
│   ├── visualize_enhanced.py      # 增强分析
│   └── start_visualization.py     # 可视化启动器
└── 📋 requirements.txt            # 依賴文件
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 安装GCBF+专用依赖
pip install -r grid_gatmf_gcbf+/gcbf_plus/requirements.txt
pip install -r grid_gatmf_gcbf+/convert/requirements.txt
```

### 2. 完整训练管道（推荐）

```bash
# 运行完整的GAT-MF → 转换 → GCBF+管道
python train_full_pipeline.py
```

这个脚本会自动完成：
- ✅ GAT-MF训练（目标80%到达率）
- ✅ PyTorch→JAX模型转换
- ✅ GCBF+训练（使用GAT-MF作为参考策略）
- ✅ 生成可视化和结果报告

### 3. 分阶段训练

如果需要分别运行各个阶段：

```bash
# 第一阶段：GAT-MF训练
python grid_gatmf_gcbf+/train_gat_enhanced.py

# 第二阶段：GCBF+训练（需要先完成第一阶段）
python train_gcbf_plus.py
```

### 4. 可视化训练过程

```bash
# 启动可视化工具
python start_visualization.py

# 或直接生成Web可视化
python visualize_web.py --episodes 3 --steps 30 --open
```

## ⚙️ 环境配置

- **环境**: 3×3网格
- **智能体**: 2個
- **障碍物**: 1個
- **连接方式**: 局部连接（相鄰格子）
- **动作空間**: 5個離散动作（停留、上、下、左、右）

## 📊 性能指標

### GAT-MF阶段
- **目標到达率**: ≥80%
- **训练回合**: 800回合
- **学习率**: 5e-4
- **探索噪声**: 0.1（逐漸衰減）

### GCBF+阶段
- **參考策略**: 转换後的GAT-MF策略
- **安全约束**: α=1.0
- **训练步數**: 2000步
- **学习率**: Actor=1e-4, CBF=1e-4

## 🛡️ 安全性增强

GCBF+通過以下機制提高安全性：
1. **控制屏障函数（CBF）**: 確保安全约束
2. **二次規劃（QP）**: 求解安全控制动作
3. **參考策略**: 使用GAT-MF提供高質量的參考控制

## 📈 结果分析

训练完成後，在 `results/` 目錄中可以找到：
- 📊 `final_report.json`: 完整实验报告
- 📁 `gat_mf/`: GAT-MF模型和日誌
- 📁 `gcbf_plus/`: GCBF+模型和日誌
- 📁 `visualizations/`: 可视化文件
- 📁 `logs/`: 詳細训练日誌

## 🔧 自定義配置

### 调整到达率目標

在 `train_full_pipeline.py` 中修改：
```python
pipeline = TrainingPipeline(
    target_arrival_rate=0.8,  # 调整目標到达率
    enable_visualization=True
)
```

### 调整训练参数

在对应的训练器中修改参数：
- **GAT-MF**: `grid_gatmf_gcbf+/train_gat_enhanced.py`
- **GCBF+**: `train_gcbf_plus.py`

## 🎨 可视化功能

### Web可视化（推薦）
- 🌐 交互式HTML界面
- 📊 智能体軌跡動畫
- 🔗 注意力连接顯示
- ⏯️ 播放控制和速度調節

### 終端可视化
- 📟 实时ASCII顯示
- 📈 训练進度图表
- 📊 关键指標监控

## 🚨 故障排除

### 常见问题

1. **模塊导入错误**
   ```bash
   # 確保在項目根目錄运行
   cd gatmf_gcbf_new-main
   python train_full_pipeline.py
   ```

2. **GAT-MF到达率不足**
   - 增加训练回合數
   - 调整学习率和探索参数
   - 检查奖励函数設計

3. **GCBF+训练失败**
   - 確保GAT-MF模型转换成功
   - 检查JAX依賴安裝
   - 降低训练步數进行测试

### 日誌查看

```bash
# 查看GAT-MF训练日誌
tail -f results/logs/gat_mf_training.json

# 查看GCBF+训练日誌  
tail -f results/logs/gcbf_plus_training.json
```

## 📚 技術細節

### 算法原理
- **GAT-MF**: 使用圖注意力機制处理智能体間交互，通過演員-評論家框架学习策略
- **GCBF+**: 集成控制屏障函数確保安全性，使用二次規劃求解最優控制

### 框架兼容
- **GAT-MF**: PyTorch实现
- **GCBF+**: JAX/Flax实现  
- **转换工具**: 無縫對接兩個框架

## 🤝 貢獻

歡迎提交Issues和Pull Requests來改进項目！

## 📄 許可證

本項目采用MIT許可證。

---

🎯 **目標**: 通過兩阶段训练实现高到达率（≥80%）和高安全率的多智能体控制
🛡️ **特色**: GAT-MF + GCBF+结合，安全性與性能並重