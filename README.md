# AlphaZero 五子棋 & Pente

## 功能特性

- **AlphaZero 算法**: 纯自对弈强化学习
- **ResNet 架构**: 带有策略头和价值头的深度残差网络
- **批量 MCTS**: 高效的蒙特卡洛树搜索，带神经网络引导
- **双游戏支持**: 统一架构支持五子棋和 Pente
- **GPU 加速**: CUDA 支持快速训练
- **GUI 界面**: PyGame 可视化用于人机对战
- **训练流程**: 完整的自对弈 → 训练 → 评估循环

## 游戏规则

### 五子棋 (Gomoku)
- **目标**: 率先连成 5 子（横、竖或斜线）
- **棋盘**: 15×15
- **玩家**: 2 人（黑子先行）

### Pente
- **目标**: 连成 5 子 **或** 吃掉 10 颗对手棋子（5 对）
- **吃子**: 用你的棋子包围 2 颗对手棋子
- **模式**: `你的 - 对手 - 对手 - 你的` 可以移除对手棋子
- **棋盘**: 15×15

***
## 工作原理

### 1. 游戏表示

**状态编码（3 个通道）：**
- 通道 0: 当前玩家位置
- 通道 1: 对手位置  
- 通道 2: 上下文平面（回合指示器）

对于 Pente，吃子从棋盘历史中隐式推断。

### 2. 神经网络

**架构：**
- 初始卷积层（3×3，128 个滤波器）
- 6 个带批归一化的残差块
- **策略头**: 输出落子概率（对 225 个动作进行 softmax）
- **价值头**: 输出局面评估（tanh，范围 [-1,1]）

**训练：**
- 优化器: Adam（lr=1e-3, weight_decay=1e-4）
- 损失函数: MSE（价值） + KL 散度（策略）
- 批次大小: 128
- 梯度裁剪: 范数 3.0

### 3. MCTS 算法

**关键特性：**
- 使用 PUCT 公式进行动作选择
- Dirichlet 噪声用于探索（仅训练时）
- 批量神经网络推理
- 带棋盘哈希的置换表

**搜索：**
每次模拟：
  1. 使用 PUCT 选择动作
  2. 用神经网络评估扩展叶子节点
  3. 沿路径回传价值

### 4. 训练流程
迭代循环：
  1. 自对弈: 使用 MCTS + 当前模型生成对局
  2. 存储: 将（状态，策略，结果）添加到回放缓冲区
  3. 训练: 采样小批次并更新网络
  4. 评估: 测试新模型与当前最佳模型的对比
  5. 接受: 如果胜率 ≥ 55%，提升为最佳模型

***
### 与 AI 对战
```bash
# 五子棋
python play.py --game gomoku --model models_gomoku/best_model.pt

# Pente
python play.py --game pente --model models_pente/best_model.pt

# 使用 GUI
python interface_pygame.py
```

### 模型评估
```bash
# 让两个模型相互对战
python play_loop.py player_alpha player_alpha2 50

# 与基线模型评估
python play_loop.py player_alpha player_mcts 50
```

***
## 团队: NinjaTurtles
- André Amaral
- Gabriel Oliveira
- José Sousa
- Simão Gomes