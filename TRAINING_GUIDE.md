# AlphaZero 五子棋训练指南

## 📋 目录

- [项目概述](#项目概述)
- [训练流程](#训练流程)
- [核心参数说明](#核心参数说明)
- [参数调优建议](#参数调优建议)
- [训练监控](#训练监控)
- [常见问题](#常见问题)
- [已修复的Bug](#已修复的bug)

---

## 🎯 项目概述

本项目使用 **AlphaZero** 算法训练五子棋（Gomoku）AI，通过自我对弈和强化学习实现从零开始学习。

### 核心特性

- ✅ **多进程自对弈**：支持多达28个并行进程
- ✅ **MCTS搜索**：基于蒙特卡洛树搜索的策略改进
- ✅ **残差神经网络**：使用ResNet架构预测策略和价值
- ✅ **经验回放**：基于缓冲区的样本复用
- ✅ **自我提升**：候选模型与最佳模型对战，优胜者晋级

### 技术栈

- **深度学习框架**：PyTorch
- **搜索算法**：蒙特卡洛树搜索（MCTS）
- **并行计算**：multiprocessing
- **游戏规则**：五子棋（15×15棋盘）

---

## 🔄 训练流程

### 完整训练循环

每次迭代包含以下阶段：

```
┌──────────────────────────────────────────────────────────┐
│                    迭代开始                               │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  阶段1: 自对弈生成 (Self-Play)                           │
│  - 使用候选模型进行自我对弈                               │
│  - 生成训练样本 (状态, 策略, 胜负)                        │
│  - 应用对称增强 (旋转、翻转)                              │
│  - 存入经验回放缓冲区                                      │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  阶段2: 神经网络训练 (Training)                           │
│  - 从缓冲区随机采样批次                                    │
│  - 训练策略头 (Policy) 和价值头 (Value)                  │
│  - 使用 Adam 优化器更新权重                               │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  阶段3: 模型评估 (Evaluation)                             │
│  - 候选模型 vs 最佳模型对战                               │
│  - 轮流先手，确保公平                                      │
│  - 随机前两手，增加开局多样性                              │
│  - 计算胜率                                               │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  阶段4: 模型更新 (Model Update)                           │
│  - 如果胜率 ≥ 阈值: 接受候选模型 → 新的最佳模型           │
│  - 如果胜率 < 阈值: 拒绝候选模型 → 从最佳模型恢复         │
│    (注意: 拒绝时不继承优化器状态，重置为初始状态)          │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│                   下一次迭代                              │
└──────────────────────────────────────────────────────────┘
```

### 阶段详解

#### 1. 自对弈阶段 (Self-Play)

**目的**：生成高质量的训练数据

**过程**：
1. 使用候选模型与自己对弈
2. 每一步记录：
   - 当前局面编码 (状态)
   - MCTS搜索得到的概率分布 (策略)
   - 从当前玩家视角的最终结果 (胜负)
3. 应用8种对称变换（4种旋转 × 2种翻转）
4. 所有样本存入经验回放缓冲区

**关键特性**：
- 前N步添加 Dirichlet 噪声增强探索
- 温度系数递减：前期探索，后期确定
- 多进程并行：大幅提升数据生成速度

#### 2. 训练阶段 (Training)

**目的**：让神经网络学习自对弈数据中的模式

**过程**：
1. 从经验回放缓冲区随机采样批次
2. 前向传播得到预测
3. 计算损失：
   - **策略损失**：KL散度 (预测策略 vs MCTS策略)
   - **价值损失**：MSE (预测价值 vs 实际胜负)
   - **总损失**：策略损失 + 价值损失
4. 反向传播更新权重

**关键技术**：
- 梯度裁剪：防止梯度爆炸
- 权重衰减：L2正则化防止过拟合
- 批量归一化：加速训练收敛

#### 3. 评估阶段 (Evaluation)

**目的**：测试候选模型是否比当前最佳模型更强

**过程**：
1. 候选模型 vs 最佳模型对战N局
2. 轮流先手（偶数局/奇数局）
3. 每局随机前两手，增加开局多样性
4. 不加探索噪声，确定性选择（argmax）
5. 计算新模型的胜率

**公平性保证**：
- ✅ 轮流先手（各50%）
- ✅ 随机开局（避免所有对局相同）
- ✅ 不加噪声（纯实力对比）

#### 4. 模型更新阶段

**接受条件**：`胜率 ≥ win_rate_threshold`

**接受时**：
- 候选模型晋级为新的最佳模型
- 保存checkpoint到磁盘
- 创建新的候选模型（从最佳模型复制）
- **继承优化器状态**（保持训练动量）

**拒绝时**：
- 保留当前最佳模型
- 候选模型从最佳模型恢复权重
- **不继承优化器状态**（重置，避免陷入局部最优）

---

## ⚙️ 核心参数说明

### 🎲 游戏参数

| 参数 | 当前值 | 说明 | 推荐范围 |
|------|--------|------|---------|
| `game_name` | `"gomoku"` | 游戏类型 | 固定 |
| `board_size` | `15` | 棋盘大小 | 9/11/15/19 |

### 🔁 训练迭代参数

| 参数 | 当前值 | 说明 | 推荐范围 |
|------|--------|------|---------|
| `num_iterations` | `100` | 总训练迭代次数 | 50-200 |
| `games_per_iteration` | `70` | 每次迭代的自对弈局数 | 50-100 |
| `next_iteration_continuation` | `1` | 起始迭代编号（用于断点续训） | ≥1 |

**说明**：
- 迭代次数越多，模型越强，但训练时间线性增长
- 每次迭代游戏数越多，样本越充足，但单次迭代时间越长

### 🌲 MCTS 参数

| 参数 | 当前值 | 说明 | 推荐范围 | 影响 |
|------|--------|------|---------|------|
| `n_simulations` | `800` | 自对弈时MCTS模拟次数 | 400-1600 | 棋力vs速度 |
| `eval_mcts_simulations` | `800` | 评估时MCTS模拟次数 | 800-2400 | 评估准确性 |
| `cpuct` | `1.0` | 探索常数 (UCB公式) | 0.5-2.0 | 探索vs利用 |
| `dirichlet_alpha` | `0.15` | Dirichlet噪声参数α | 0.1-0.3 | 探索强度 |
| `epsilon` | `0.15` | 噪声混合比例 | 0.15-0.25 | 探索比例 |
| `apply_dirichlet_n_first_moves` | `15` | 前N步加噪声 | 10-20 | 开局探索 |

**MCTS 参数详解**：

#### `n_simulations` (模拟次数)
- **作用**：每一步思考的深度
- **权衡**：
  - 更多模拟 → 棋力更强，但速度更慢
  - 更少模拟 → 速度更快，但棋力较弱
- **建议**：
  - 早期训练（1-30轮）：400-800
  - 中期训练（31-70轮）：800-1200
  - 后期训练（71+轮）：1200-1600

#### `cpuct` (探索常数)
- **作用**：控制UCB公式中的探索倾向
- **公式**：`UCB = Q + cpuct × P × √(N_parent) / (1 + N_child)`
- **影响**：
  - `cpuct` 较大 → 更倾向探索未尝试的动作
  - `cpuct` 较小 → 更倾向利用已知的好动作
- **建议**：1.0-1.5（五子棋推荐1.0-1.2）

#### `dirichlet_alpha` (Dirichlet α)
- **作用**：控制探索噪声的分布形态
- **计算**：推荐公式 `α = 10 / 平均合法动作数`
- **五子棋特性**：
  - 开局：200+合法动作 → α ≈ 0.05
  - 中局：100-150合法动作 → α ≈ 0.10
  - 综合建议：**0.15-0.3**
- **影响**：
  - α 较小 → 噪声集中在少数动作（探索不足）
  - α 较大 → 噪声分散（探索充分）

#### `epsilon` (噪声混合比例)
- **作用**：控制噪声在最终策略中的占比
- **公式**：`P_final = (1-ε) × P_network + ε × Noise`
- **AlphaZero标准**：0.25（25%噪声）
- **当前配置**：0.15（保守，适合早期训练）
- **建议**：早期0.25，后期可降至0.15

### 🎓 神经网络参数

| 参数 | 当前值 | 说明 | 推荐范围 |
|------|--------|------|---------|
| `n_res_blocks` | `3` | 残差块数量 | 3-10 |
| `channels` | `64` | 卷积通道数 | 64-256 |
| `lr` | `0.001` | 学习率 | 1e-4 ~ 1e-3 |
| `weight_decay` | `0.0001` | 权重衰减 (L2正则) | 1e-5 ~ 1e-3 |

**网络结构说明**：
```
输入 (3, 15, 15)
    ↓
初始卷积 (3→64通道)
    ↓
3个残差块 (64通道)
    ↓
    ├─→ 策略头 → (225,) logits
    └─→ 价值头 → (1,) value ∈ [-1,1]
```

### 💾 训练参数

| 参数 | 当前值 | 说明 | 推荐范围 |
|------|--------|------|---------|
| `buffer_size` | `60000` | 经验回放缓冲区大小 | 20000-100000 |
| `batch_size` | `128` | 训练批次大小 | 64-256 |
| `epochs_per_iter` | `3` | 每次迭代训练轮数 | 2-5 |

**缓冲区大小计算**：
```
每局游戏样本数 ≈ 平均步数 × 8(对称) ≈ 40 × 8 = 320
每次迭代样本数 ≈ 70局 × 320 = 22,400
缓冲区可容纳 ≈ 60000 / 22400 ≈ 2.7次迭代
```

**建议**：容纳2-5次迭代的样本量

### 🎯 评估参数

| 参数 | 当前值 | 说明 | 推荐范围 |
|------|--------|------|---------|
| `eval_games` | `50` | 评估对局数 | 20-100 |
| `win_rate_threshold` | `0.52` | 接受阈值 | 0.50-0.55 |

**阈值选择**：
- `0.50`：等价于随机（太宽松）
- `0.52`：略有提升即接受（当前配置，**推荐**）
- `0.55`：明显提升才接受（较严格）
- `0.60`：大幅提升才接受（过于严格，可能拒绝有效改进）

**评估局数 vs 统计显著性**：
```
20局：胜率标准差 ≈ 11% (波动大)
50局：胜率标准差 ≈ 7%  (较稳定) ← 当前配置
100局：胜率标准差 ≈ 5% (很稳定，但耗时)
```

### 🖥️ 多进程参数

| 参数 | 当前值 | 说明 | 推荐范围 |
|------|--------|------|---------|
| `selfplay_num_workers` | `28` | 自对弈进程数 | CPU核心数-1 |
| `selfplay_device` | `"cpu"` | 自对弈设备 | `"cpu"` |
| `selfplay_games_per_task` | `1` | 每个任务的游戏数 | 1-2 |
| `selfplay_torch_threads` | `1` | 每进程PyTorch线程数 | 1 |
| `eval_num_workers` | `28` | 评估进程数 | CPU核心数-1 |
| `eval_device` | `"cpu"` | 评估设备 | `"cpu"` |

**多进程注意事项**：
- ⚠️ Windows 使用 `spawn` 模式（每个子进程重新import）
- ⚠️ GPU多进程需要谨慎（建议用CPU避免CUDA争用）
- ✅ 进程数 = CPU核心数-1（留一个核心给主进程）
- ✅ 每进程1个PyTorch线程（避免CPU过度竞争）

### 🌡️ 温度参数

| 参数 | 当前值 | 说明 | 影响 |
|------|--------|------|------|
| `temp_threshold` | `10` | 温度衰减阈值 | 探索vs确定性 |

**温度调度公式**：
```python
temperature = max(0.0, 1.0 - move_number / temp_threshold)
```

**示例**（threshold=10）：
- 第1步：temp = 0.9 → 高探索
- 第5步：temp = 0.5 → 中等探索
- 第10步：temp = 0.0 → 完全确定（argmax）
- 第11+步：temp = 0.0 → 完全确定

**作用**：
- 前期：高温度 → 探索多样性
- 后期：低温度 → 确定性选择最优动作

---

## 📊 参数调优建议

### 🎯 目标导向的参数配置

#### 目标1：快速验证（原型阶段）

```python
train_alphazero(
    num_iterations=20,           # 少量迭代
    games_per_iteration=30,      # 少量对局
    n_simulations=200,           # 低模拟次数
    eval_games=20,               # 快速评估
    buffer_size=20000,
    epochs_per_iter=2,
    selfplay_num_workers=8,      # 适中进程数
)
```

**预期**：
- 单次迭代：5-10分钟
- 20轮总耗时：2-3小时
- 棋力：能战胜随机玩家

---

#### 目标2：平衡训练（推荐，当前配置）

```python
train_alphazero(
    num_iterations=100,          # 充足迭代
    games_per_iteration=70,      # 充足样本
    n_simulations=800,           # 中等模拟
    eval_games=50,               # 稳定评估
    buffer_size=60000,
    epochs_per_iter=3,
    selfplay_num_workers=28,     # 高并行
)
```

**预期**：
- 单次迭代：20-40分钟（取决于硬件）
- 100轮总耗时：40-60小时
- 棋力：业余中等水平（可能击败人类初学者）

---

#### 目标3：顶级棋力（竞赛级别）

```python
train_alphazero(
    num_iterations=200,          # 大量迭代
    games_per_iteration=100,     # 大量样本
    n_simulations=1600,          # 高模拟次数
    eval_games=100,              # 高置信度评估
    eval_mcts_simulations=2400,  # 评估更谨慎
    buffer_size=100000,
    epochs_per_iter=5,
    n_res_blocks=6,              # 更深网络
    channels=128,                # 更宽网络
    selfplay_num_workers=32,
)
```

**预期**：
- 单次迭代：1-2小时
- 200轮总耗时：200-400小时（1-2周）
- 棋力：接近业余高手

---

### 🔧 常见调优场景

#### 场景1：训练不稳定（胜率波动大）

**可能原因**：
- 评估样本太少
- 模拟次数太少
- 学习率过高

**解决方案**：
```python
eval_games=100,              # 增加评估局数
n_simulations=1200,          # 增加模拟次数
lr=5e-4,                     # 降低学习率
win_rate_threshold=0.54,     # 提高接受阈值
```

---

#### 场景2：训练速度太慢

**可能原因**：
- 模拟次数过高
- 进程数不足
- 每次迭代游戏数过多

**解决方案**：
```python
n_simulations=400,           # 降低模拟次数
games_per_iteration=50,      # 减少游戏数
selfplay_num_workers=32,     # 增加并行度
selfplay_games_per_task=2,   # 每任务多局（减少进程间通信）
```

---

#### 场景3：模型棋力提升缓慢

**可能原因**：
- 探索不足
- 网络容量不足
- 训练轮数不够

**解决方案**：
```python
epsilon=0.25,                # 增加探索噪声
dirichlet_alpha=0.3,         # 增强探索
n_res_blocks=6,              # 增加网络深度
channels=128,                # 增加网络宽度
epochs_per_iter=5,           # 增加训练轮数
```

---

#### 场景4：过拟合（训练损失低但棋力不提升）

**可能原因**：
- 缓冲区太小（样本重复训练）
- 正则化不足

**解决方案**：
```python
buffer_size=100000,          # 增大缓冲区
weight_decay=5e-4,           # 增加L2正则
epochs_per_iter=2,           # 减少训练轮数（每次迭代）
```

---

### 📈 动态参数调整策略

#### 阶段式训练（推荐）

```python
# 阶段1：快速探索（1-30轮）
n_simulations=400
epsilon=0.25
games_per_iteration=50

# 阶段2：稳定提升（31-70轮）
n_simulations=800
epsilon=0.20
games_per_iteration=70

# 阶段3：精细优化（71-100轮）
n_simulations=1200
epsilon=0.15
games_per_iteration=80
eval_mcts_simulations=1600
```

**实现方式**：
1. 训练前30轮，暂停
2. 修改参数，使用 `pretrained_model_path` 和 `next_iteration_continuation=31` 继续
3. 重复上述步骤

---

## 📊 训练监控

### 关键指标

#### 1. 自对弈胜负统计

```
自对弈完成：耗时 45.2s，胜负统计={0: 5, 1: 32, 2: 33}，buffer_size=22400
```

**解读**：
- `0: 5` → 平局5局（正常，五子棋平局罕见）
- `1: 32` → 玩家1胜32局
- `2: 33` → 玩家2胜33局
- **理想**：两个玩家胜局接近（说明自对弈公平）
- **异常**：如果某一方胜局 >70%，可能有bug

#### 2. 训练损失

```
epoch 1/3 finished in 15.3s, last_loss={'policy_loss': 2.354, 'value_loss': 0.567, 'total_loss': 2.921}
```

**正常趋势**：
- 早期迭代（1-10）：`total_loss` 约 3-5
- 中期迭代（11-50）：`total_loss` 约 1-3
- 后期迭代（51+）：`total_loss` 约 0.5-1.5

**警告信号**：
- 损失突然上升 → 可能学习率过高或训练不稳定
- 损失不下降 → 可能陷入局部最优或学习率过低
- 损失为 NaN → 梯度爆炸（需要降低学习率或检查数据）

#### 3. 评估胜率

```
评估完成：耗时 85.1s，胜率=0.560（28/50），平局=3
```

**解读**：
- `胜率=0.560` → 候选模型胜率56%
- `28/50` → 50局中胜28局
- `平局=3` → 3局平局

**期望曲线**：
```
迭代1-10：  胜率 50-55% (缓慢提升)
迭代11-30： 胜率 52-58% (加速提升)
迭代31-60： 胜率 53-60% (稳定提升)
迭代61+：   胜率 52-56% (微小提升，接近收敛)
```

#### 4. 接受/拒绝率

**理想比例**：
- 前30轮：接受率 50-70%
- 中期：接受率 40-60%
- 后期：接受率 20-40%（收敛后拒绝率上升是正常的）

**异常情况**：
- 接受率 >90%：阈值过低或训练过快（可能不稳定）
- 接受率 <10%：阈值过高或训练停滞

---

### 日志示例

```
=== ITER 15/99: 自对弈生成 (games=70, sims=800) ===
自对弈完成：耗时 52.3s，胜负统计={0: 4, 1: 33, 2: 33}，buffer_size=45600

Training candidate model: buffer=45600, batch_size=128, epochs_per_iter=3
  epoch 1/3 finished in 14.2s, last_loss={'policy_loss': 1.845, 'value_loss': 0.423, 'total_loss': 2.268}
  epoch 2/3 finished in 14.1s, last_loss={'policy_loss': 1.782, 'value_loss': 0.401, 'total_loss': 2.183}
  epoch 3/3 finished in 14.3s, last_loss={'policy_loss': 1.731, 'value_loss': 0.389, 'total_loss': 2.120}

评估完成：耗时 78.5s，胜率=0.580（29/50），平局=2
 候选模型被接受 -> 提升为最佳模型。
 Saved snapshot: models/snapshot_iter15_20260108_143022.pt

迭代 15 完成，耗时 159.4s。本次迭代获胜者: {0: 4, 1: 33, 2: 33}
```

---

### 可视化监控（可选）

可以使用以下脚本生成训练曲线：

```python
import matplotlib.pyplot as plt

# 从日志提取数据
iterations = [1, 2, 3, ..., 100]
win_rates = [0.52, 0.54, 0.56, ...]
losses = [3.2, 2.8, 2.5, ...]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(iterations, win_rates)
plt.xlabel('Iteration')
plt.ylabel('Win Rate')
plt.title('Evaluation Win Rate')

plt.subplot(1, 2, 2)
plt.plot(iterations, losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.tight_layout()
plt.savefig('training_curves.png')
```

---

## ❓ 常见问题

### Q1: 第一轮迭代就被拒绝，正常吗？

**A**: 现在不正常了（已修复bug）。

**修复前**：第一轮大概率被拒绝，因为 `model_best` 和 `model_candidate` 是独立随机初始化的。

**修复后**：第一轮应该有较高概率被接受（60-80%），因为候选模型经过训练应该比随机初始化的最佳模型更强。

如果修复后第一轮仍被拒绝：
- 检查训练样本是否足够（buffer_size）
- 检查训练轮数（epochs_per_iter）
- 可能需要降低 `win_rate_threshold`（如0.50）

---

### Q2: 评估时所有对局都一样，怎么办？

**A**: 已修复！现在随机前两手。

**原因**：评估时使用确定性选择（argmax）且不加噪声，如果开局相同，所有对局会完全一样。

**修复**：每局随机前两手，然后从第三手开始真正评估。

---

### Q3: 训练速度太慢，怎么加速？

**A**: 多方面优化：

1. **降低模拟次数**：`n_simulations=400`
2. **增加并行度**：`selfplay_num_workers=CPU核心数-1`
3. **减少每次迭代游戏数**：`games_per_iteration=50`
4. **使用GPU**（如果有）：修改 `network.py` 中的 `device`
5. **减少评估局数**：`eval_games=20`（但会降低评估准确性）

---

### Q4: 如何断点续训？

**A**: 三步走：

```python
# 1. 设置预训练模型路径
pretrained_model_path="models/model_best_iter50_20260108_120000.pt"

# 2. 设置起始迭代
next_iteration_continuation=51  # 从第51轮开始

# 3. 运行训练
train_alphazero(...)
```

---

### Q5: 如何测试训练好的模型？

**A**: 使用对战脚本（需要创建）：

```python
from network import PyTorchModel
from mcts.new_mcts_alpha import MCTS
from games.gomoku import Gomoku

# 加载模型
model = PyTorchModel(board_size=15, action_size=225)
model.load("models/model_best_iter100_xxx.pt")

# 创建游戏
game = Gomoku(size=15)
mcts = MCTS(
    game_class=Gomoku,
    n_simulations=1600,  # 更多模拟 = 更强棋力
    nn_model=model,
    cpuct=1.0,
    add_dirichlet_noise=False
)

# 人机对战循环
while not game.is_game_over():
    game.display()
    
    if game.current_player == 1:  # AI
        pi = mcts.run(game, len(game.move_history))
        action = int(np.argmax(pi))
    else:  # 人类
        print("你的回合，输入坐标 (如: 7 7):")
        r, c = map(int, input().split())
        action = r * 15 + c
    
    r, c = divmod(action, 15)
    game.do_move((r, c))

print(f"游戏结束！获胜者: {game.get_winner()}")
```

---

### Q6: 训练到多少轮可以停止？

**A**: 观察收敛信号：

**明显收敛**：
- 连续10轮评估胜率在 50-52% 之间波动
- 训练损失不再下降
- 接受率 <20%

**何时停止**：
- 目标达成（如能击败特定对手）
- 胜率提升 <1% 连续20轮
- 训练损失曲线完全平坦

**典型轮数**：
- 入门级：30-50轮
- 中级：70-100轮
- 高级：150-200轮

---

### Q7: 模型保存在哪里？

**A**: `models/` 目录

**文件类型**：
1. **接受时保存**：`model_best_iter{N}_{timestamp}.pt`
   - 例：`model_best_iter15_20260108_143022.pt`
2. **定期快照**：`snapshot_iter{N}_{timestamp}.pt`
   - 由 `save_every` 参数控制

**加载模型**：
```python
model = PyTorchModel(board_size=15, action_size=225)
model.load("models/model_best_iter100_xxx.pt")
```

---

### Q8: 内存不足怎么办？

**A**: 减少内存占用：

1. **减小缓冲区**：`buffer_size=30000`
2. **减小批次**：`batch_size=64`
3. **减少并行进程**：`selfplay_num_workers=8`
4. **减少网络容量**：
   ```python
   n_res_blocks=2
   channels=32
   ```

---

### Q9: 如何调整让训练更稳定？

**A**: 稳定性优先配置：

```python
train_alphazero(
    # 增加评估样本
    eval_games=100,
    eval_mcts_simulations=1200,
    
    # 提高接受阈值
    win_rate_threshold=0.54,
    
    # 降低学习率
    lr=5e-4,
    
    # 增加正则化
    weight_decay=5e-4,
    
    # 充足训练
    epochs_per_iter=4,
)
```

---

## 🐛 已修复的Bug

### Bug 1: 初始化不一致（已修复）

**问题**：
```python
# 修复前
model_best = PyTorchModel()      # 随机权重A
model_candidate = PyTorchModel()  # 随机权重B (不同！)
```

**影响**：第一轮评估实际上是两个不同的随机模型对战，胜率随机，大概率被拒绝。

**修复**：
```python
# 修复后
model_best = PyTorchModel()
model_candidate = PyTorchModel()
model_candidate.net.load_state_dict(model_best.net.state_dict())  # 复制权重
```

**修复位置**：`train.py` 第553-554行

---

### Bug 2: 迭代次数多1轮（已修复）

**问题**：
```python
# 修复前
for it in range(start, start + num_iterations + 1):  # 多加了1
```

**影响**：配置100轮，实际执行102轮（浪费时间）。

**修复**：
```python
# 修复后
for it in range(start, start + num_iterations):  # 正确
```

**修复位置**：`train.py` 第567行

---

### Bug 3: 硬编码棋盘大小（已修复）

**问题**：
```python
# 修复前
r = random.randint(0, 14)  # 硬编码14
c = random.randint(0, 14)
```

**影响**：仅支持15×15棋盘，其他尺寸会出错。

**修复**：
```python
# 修复后
r = random.randint(0, board_size - 1)
c = random.randint(0, board_size - 1)
```

**修复位置**：`train.py` 评估函数多处

---

### Bug 4: 评估逻辑反转（已修复）

**问题**：
```python
# 修复前
game.do_move((r, c))  # 随机第一手，current_player 切换为 2
new_starts = (i % 2 == 0)  # 预期：新模型先手
# 但实际：best 模型先走第二步（逻辑反了！）
```

**影响**：
1. `new_starts=True` 时，实际是 best 模型先手
2. 只随机一手导致评估开局重复

**修复**：
```python
# 修复后
# 随机前两手
game.do_move((r1, c1))  # 第一手
game.do_move((r2, c2))  # 第二手
# 现在 current_player = 1，逻辑正确
new_starts = (i % 2 == 0)
```

**修复位置**：`train.py` 第186-201行, 第377-393行

---

### Bug 5: MCTS参数不当（已修复）

**问题**：
```python
# 修复前
dirichlet_alpha=0.03  # 太小（围棋参数）
epsilon=0.03          # 太小（仅3%探索）
apply_dirichlet_n_first_moves=10  # 偏少
```

**影响**：
- 探索严重不足
- 开局变化少
- 容易陷入局部最优

**修复**：
```python
# 修复后（方案A：保守优化）
dirichlet_alpha=0.15  # 适配五子棋
epsilon=0.15          # 增强探索
apply_dirichlet_n_first_moves=15  # 覆盖开局
```

**修复位置**：`train.py` MCTS实例化处（多处）

---

### Bug 6: 优化器状态不当继承（已修复）

**问题**：
```python
# 修复前：候选模型被拒绝时
model_candidate.optimizer.load_state_dict(model_best.optimizer.state_dict())
# 继承了优化器状态（可能导致陷入局部最优）
```

**影响**：被拒绝的模型说明训练方向不对，不应该继承其优化器的动量。

**修复**：
```python
# 修复后：被拒绝时
model_candidate = PyTorchModel(...)
model_candidate.net.load_state_dict(model_best.net.state_dict())
# 不加载优化器状态，重置为初始状态
```

**修复位置**：`train.py` 第740-744行

---

### Bug 7: 评估时冗余参数（已修复）

**问题**：
```python
# 修复前：评估时
mcts = MCTS(
    add_dirichlet_noise=False,  # 不加噪声
    dirichlet_alpha=0.03,       # 但设置了这些参数（冗余）
    epsilon=0.03,
    apply_dirichlet_n_first_moves=10,
)
```

**影响**：无实际影响，但降低代码可读性。

**修复**：
```python
# 修复后：评估时
mcts = MCTS(
    add_dirichlet_noise=False,  # 只保留必要参数
    # 省略其他 Dirichlet 参数
)
```

**修复位置**：`train.py` 评估MCTS实例化处

---

## 📚 参考资源

### 论文
- **AlphaGo Zero**: [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- **AlphaZero**: [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://arxiv.org/abs/1712.01815)

### 代码实现
- [Alpha Zero General (GitHub)](https://github.com/suragnair/alpha-zero-general)
- [LeelaZero (围棋)](https://github.com/leela-zero/leela-zero)

### 相关资源
- [MCTS详解](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [残差网络 (ResNet)](https://arxiv.org/abs/1512.03385)
- [Dirichlet分布](https://en.wikipedia.org/wiki/Dirichlet_distribution)

---

## 📞 支持与贡献

### 遇到问题？

1. 检查本文档的"常见问题"部分
2. 查看训练日志中的错误信息
3. 确认参数配置是否合理
4. 检查硬件资源（内存、CPU）

### 改进建议

欢迎提出改进建议：
- 参数调优经验
- 新的训练策略
- Bug修复
- 文档改进

---

## 📄 版本历史

- **v1.0** (2026-01-08): 初始版本，包含所有bug修复和参数优化
  - 修复7个关键bug
  - 优化MCTS参数
  - 完善评估逻辑
  - 添加详细文档

---

## 🎉 开始训练！

现在你已经掌握了所有必要的知识，可以开始训练了：

```bash
python train.py
```

祝训练顺利！🚀

---

**最后更新**: 2026年1月8日
**文档维护者**: AI Assistant
**项目**: AlphaZero Gomoku
