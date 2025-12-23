# AlphaZero Gomoku & Pente

## Features

- **AlphaZero Algorithm**: Pure self-play reinforcement learning
- **ResNet Architecture**: Deep residual network with policy and value heads
- **Batched MCTS**: Efficient Monte Carlo Tree Search with neural network guidance
- **Dual Game Support**: Unified architecture for Gomoku and Pente
- **GPU Accelerated**: CUDA support for fast training
- **GUI Interface**: PyGame visualization for human play
- **Training Pipeline**: Complete self-play → train → evaluate loop

## Game Rules

### Gomoku
- **Objective**: First to align 5 stones (horizontal, vertical, or diagonal)
- **Board**: 15×15
- **Players**: 2 (Black starts)

### Pente
- **Objective**: 5 in a row **OR** capture 10 opponent stones (5 pairs)
- **Captures**: Surrounding 2 opponent stones with your stones
- **Pattern**: `Your - Opp - Opp - Your` removes opponent stones
- **Board**: 15×15

***
## How It Works

### 1. Game Representation

**State Encoding (3 channels):**
- Channel 0: Current player positions
- Channel 1: Opponent positions  
- Channel 2: Context plane (turn indicator)

For Pente, captures are inferred implicitly from board history.

### 2. Neural Network

**Architecture:**
- Initial conv layer (3×3, 128 filters)
- 6 residual blocks with batch normalization
- **Policy Head**: Outputs move probabilities (softmax over 225 actions)
- **Value Head**: Outputs position evaluation (tanh, range [-1,1])

**Training:**
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Loss: MSE (value) + KL divergence (policy)
- Batch size: 128
- Gradient clipping: norm 3.0

### 3. MCTS Algorithm

**Key Features:**
- PUCT formula for action selection
- Dirichlet noise for exploration (training only)
- Batched neural network inference
- Transposition table with board hashing

**Search:**
For each simulation:
  1. Select action using PUCT
  2. Expand leaf node with NN evaluation
  3. Backup value through path

### 4. Training Pipeline
Iteration loop:
  1. Self-play: Generate games using MCTS + current model
  2. Store: Add (state, policy, outcome) to replay buffer
  3. Train: Sample minibatches and update network
  4. Evaluate: Test new model vs. current best
  5. Accept: If win rate ≥ 55%, promote to best

***
### Play Against AI
```bash
# Gomoku
python play.py --game gomoku --model models_gomoku/best_model.pt

# Pente
python play.py --game pente --model models_pente/best_model.pt

# With GUI
python interface_pygame.py
```

### Model Evaluation
```bash
# Pit two models against each other
python play_loop.py player_alpha player_alpha2 50

# Evaluate against baselines
python play_loop.py player_alpha player_mcts 50
```

***
## Team: NinjaTurtles
- André Amaral
- Gabriel Oliveira
- José Sousa
- Simão Gomes