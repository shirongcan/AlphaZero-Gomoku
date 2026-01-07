import os
import time
import random
from collections import deque
from typing import List, Tuple, Optional
import numpy as np
from network import PyTorchModel
# from mcts.mcts_alpha import MCTS
# from mcts.mcts_alpha_with_noise import MCTS
from mcts.new_mcts_alpha import MCTS
from players.player_alpha import Player
from games.gomoku import Gomoku as GameClass
from games.pente import Pente as GameClass
from datetime import datetime
from copy import deepcopy
import sys
import gc

# 在循环内动态映射游戏名称到类

# -------------------------
#  工具函数
# -------------------------
def softmax_temperature(pi: np.ndarray, temp: float) -> np.ndarray:
    if temp <= 0:
        return pi
    logits = np.log(pi + 1e-15)
    logits = logits / temp
    exps = np.exp(logits - np.max(logits))
    p = exps / np.sum(exps)
    return p


def sample_action_from_pi(pi: np.ndarray, temp: float) -> int:
    if temp == 0:
        return int(np.argmax(pi))
    p = softmax_temperature(pi, temp)
    return int(np.random.choice(len(p), p=p))


# -------------------------
#  经验回放缓冲区
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        添加示例列表 (state_enc, pi, z)
        state_enc: (C,H,W)
        pi: (action_size,)
        z: 标量 (-1,0,1)
        """
        for ex in examples:
            self.buffer.append(ex)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, k=batch_size)
        states, pis, zs = zip(*batch)
        states = np.stack(states, axis=0).astype(np.float32)
        pis = np.stack(pis, axis=0).astype(np.float32)
        zs = np.array(zs, dtype=np.float32).reshape(-1, 1)
        return states, pis, zs

    def __len__(self):
        return len(self.buffer)


# -------------------------
#  自对弈单局游戏
# -------------------------
def play_game_and_collect(mcts: MCTS, game, temp_fn, max_moves=225, use_symmetries=True):
    """
    进行一局完整的游戏并返回增强后的示例：
    final_examples: (state_enc (C,H,W), pi (A,), z 标量) 的列表
    winner: 0/1/2
    """
    examples = []
    move_number = 0

    while True:
        state_enc = game.get_encoded_state()  # 期望是视角不变的
        pi = mcts.run(game, len(game.move_history))  # 向量 (action_size,)
        pi_for_store = pi.copy()

        temp = temp_fn(move_number)
        action = sample_action_from_pi(pi, temp)

        # 安全回退：如果选择的动作不合法，使用 argmax
        valid_mask = game.get_valid_moves()
        if valid_mask[action] != 1.0:
            action = int(np.argmax(pi))
        # 存储 (state, pi, player)
        examples.append((state_enc, pi_for_store, int(game.current_player)))

        # 执行移动
        r, c = divmod(action, game.size)
        game.do_move((r, c))

        move_number += 1

        if game.is_game_over() or move_number >= max_moves:
            break

    winner = game.get_winner()  # 0/1/2

    # 将示例转换为 (state_aug, pi_aug, z)
    final_examples = []
    for state_enc, pi_vec, player in examples:
        if winner == 0:
            z = 0.0
        else:
            z = 1.0 if winner == player else -1.0

        if use_symmetries:
            syms = mcts.symmetries(state_enc, pi_vec)
            for s_aug, pi_aug in syms:
                final_examples.append((s_aug.astype(np.float32), pi_aug.astype(np.float32), z))
        else:
            final_examples.append((state_enc.astype(np.float32), pi_vec.astype(np.float32), z))

    return final_examples, winner


# -------------------------
#  模型间评估
# -------------------------
def evaluate_models(model_new: PyTorchModel,
                model_best: PyTorchModel,
                game_name: str,
                n_games: int = 20,
                n_simulations: int = 100,
                cpuct: float = 1.0) -> Tuple[float, int]:
    """
    在 model_new 和 model_best 之间进行 n_games 局游戏（轮流先手）。
    返回 (win_rate_of_new, draws)
    """
    # 选择游戏类
    if game_name.lower().startswith("pente"):
        rules_name = "pente"
    else:
        rules_name = "gomoku"

    new_wins = 0
    draws = 0
    total = n_games

    for i in range(n_games):
        game = GameClass(size=model_new.board_size)

        r = random.randint(0, 14)
        c = random.randint(0, 14)
        game.do_move((r, c))

        # 确定谁先手：新模型在偶数局先手
        new_starts = (i % 2 == 0)
        move_number = 1

        # 为两个玩家创建 MCTS 实例（扩展时各自使用自己的模型）
        mcts_new = MCTS(game_class=GameClass, n_simulations=n_simulations, nn_model=model_new, cpuct=cpuct, dirichlet_alpha=0.03, epsilon=0.03, apply_dirichlet_n_first_moves=10, add_dirichlet_noise=False)
        mcts_best = MCTS(game_class=GameClass, n_simulations=n_simulations, nn_model=model_best, cpuct=cpuct, dirichlet_alpha=0.03, epsilon=0.03, apply_dirichlet_n_first_moves=10, add_dirichlet_noise=False)

        while not game.is_game_over():
            # 根据当前玩家和谁先手决定谁下棋
            if (game.current_player == 1 and new_starts) or (game.current_player == 2 and not new_starts):
                pi = mcts_new.run(game, len(game.move_history))
            else:
                pi = mcts_best.run(game, len(game.move_history))

            # 确定性选择 (argmax)
            action = int(np.argmax(pi))
            r, c = divmod(action, game.size)
            game.do_move((r, c))
            move_number += 1
            if move_number > game.size * game.size:
                break

        winner = game.get_winner()
        if winner == 0:
            draws += 1
        else:
            # 确定新模型是否获胜
            if (winner == 1 and new_starts) or (winner == 2 and not new_starts):
                new_wins += 1

        mcts_new.clear_tree()
        mcts_best.clear_tree()
        del mcts_new
        del mcts_best
        gc.collect()

    win_rate = new_wins / float(total)
    return new_wins, win_rate, draws


# -------------------------
#  主训练循环
# -------------------------
def train_alphazero(
    game_name: str = "gomoku",
    board_size: int = 15,
    num_iterations: int = 5,
    games_per_iteration: int = 8,
    n_simulations: int = 50,
    buffer_size: int = 10000,
    batch_size: int = 128,
    epochs_per_iter: int = 2,
    temp_threshold: int = 8,
    eval_games: int = 12,
    eval_mcts_simulations: int = 200,
    win_rate_threshold: float = 0.55,
    cpuct: float = 1.2,
    model_dir: str = "models",
    save_every: int = 1,
    pretrained_model_path: Optional[str] = None,  # 新参数，用于传递预训练模型
    next_iteration_continuation: int = 1
):
    """
    核心训练流程。
    """
    os.makedirs(model_dir, exist_ok=True)

    # 根据 board_size 计算动作空间大小
    action_size = board_size * board_size  # 对于 Gomoku（或 Pente），动作是棋盘上的位置

    # 检查是否存在预训练模型
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"从以下路径加载预训练模型: {pretrained_model_path}")
        model_best = PyTorchModel(board_size=board_size, action_size=action_size)
        model_best.load(pretrained_model_path)  # 加载预训练模型
        model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
        model_candidate.net.load_state_dict(model_best.net.state_dict())
        print("预训练模型加载成功。")
    else:
        print("未找到预训练模型。初始化新模型。")
        model_best = PyTorchModel(board_size=board_size, action_size=action_size)
        model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)

    # 经验回放缓冲区
    buffer = ReplayBuffer(capacity=buffer_size)

    # 温度调度
    def temp_fn(move_number: int):
        return max(0.0, 1.0 - move_number / temp_threshold)

    for it in range(next_iteration_continuation, next_iteration_continuation + num_iterations + 1):
        t0 = time.time()
        print(f"\n=== ITER {it}/{next_iteration_continuation + num_iterations}: 自对弈生成 (games={games_per_iteration}, sims={n_simulations}) ===")

        # 使用候选模型进行自对弈生成
        winners = {0: 0, 1: 0, 2: 0}
        for g in range(games_per_iteration):
            mcts_play = MCTS(game_class=GameClass, n_simulations=n_simulations, nn_model=model_candidate, cpuct=cpuct, dirichlet_alpha=0.03, epsilon=0.03, apply_dirichlet_n_first_moves=10, add_dirichlet_noise=True)
            game = GameClass(size=board_size)
            game.current_player = 1
            examples, winner = play_game_and_collect(mcts_play, game, temp_fn, max_moves=board_size * board_size, use_symmetries=True)
            buffer.add(examples)
            winners[winner] = winners.get(winner, 0) + 1
            print(f"  gen game {g+1}/{games_per_iteration} -> winner={winner}, buffer_size={len(buffer)}")

            mcts_play.clear_tree()
            del mcts_play
            gc.collect()

        # 如果有足够的样本，训练候选模型
        if len(buffer) >= batch_size:
            print(f"\nTraining candidate model: buffer={len(buffer)}, batch_size={batch_size}, epochs_per_iter={epochs_per_iter}")
            n_batches = max(1, len(buffer) // batch_size)
            for epoch in range(epochs_per_iter):
                epoch_t0 = time.time()
                for b in range(n_batches):
                    states_b, pis_b, zs_b = buffer.sample(batch_size)
                    loss_info = model_candidate.train_batch(states_b, pis_b, zs_b, epochs=1)
                epoch_t1 = time.time()
                print(f"  epoch {epoch+1}/{epochs_per_iter} finished in {epoch_t1 - epoch_t0:.1f}s, last_loss={loss_info}")
        else:
            print(f"训练样本不足 (buffer={len(buffer)}, 需要 {batch_size})。跳过本次迭代的训练。")

        # 评估
        print("\nEvaluating candidate vs best...")
        try:
            new_wins, win_rate, draws = evaluate_models(model_candidate, model_best, game_name, n_games=eval_games, n_simulations=eval_mcts_simulations, cpuct=cpuct)
        except Exception as e:
            print("Evaluation failed (exception):", e)
            win_rate, draws = 0.0, 0

        print(f" 候选模型胜率 = {win_rate:.3f} ({new_wins}/{eval_games}) (平局={draws})")

        # 接受/拒绝
        if win_rate >= win_rate_threshold:
            print(" 候选模型被接受 -> 提升为最佳模型。")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(model_dir, f"model_best_iter{it}_{timestamp}.pt")
            model_candidate.save(path)
            model_best = model_candidate
            # 从最佳模型创建新的候选模型
            model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
            model_candidate.net.load_state_dict(model_best.net.state_dict())
            model_candidate.optimizer.load_state_dict(model_best.optimizer.state_dict())
        else:
            print(" 候选模型被拒绝 -> 从最佳模型恢复候选模型。")
            # 从最佳模型权重重置候选模型
            model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
            model_candidate.net.load_state_dict(model_best.net.state_dict())
            model_candidate.optimizer.load_state_dict(model_best.optimizer.state_dict())

        # 定期保存最佳模型状态
        if it % save_every == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(model_dir, f"snapshot_iter{it}_{timestamp}.pt")
            model_best.save(snapshot_path)
            print(f" Saved snapshot: {snapshot_path}")

        t1 = time.time()
        print(f"迭代 {it} 完成，耗时 {(t1 - t0):.1f}s。本次迭代获胜者: {winners}")

    print("\n=== 训练完成 ===")

# -------------------------
#  入口点
# -------------------------
if __name__ == "__main__":
    train_alphazero(
        game_name="gomoku",           # 游戏 Gomoku
        board_size=15,                # 棋盘大小 (15x15)

        num_iterations=30,           # 30 次训练迭代
        games_per_iteration=40,       # 每次迭代 40 局游戏

        n_simulations=2000,          # MCTS 2000 次模拟
        cpuct=1.0,                   # MCTS 的探索/利用平衡因子

        buffer_size=80000,           # 经验回放缓冲区，最多 80,000 个样本（容纳约5次迭代）
        batch_size=128,               # 每个训练批次 128 个样本
        epochs_per_iter=5,           # 每次迭代 5 个训练轮次（GPU快时可增加）

        temp_threshold=10,           # 探索温度阈值
        eval_games=30,               # 30 局评估游戏（提高统计稳定性）
        eval_mcts_simulations=2000,  # 评估时 MCTS 2000 次模拟
        win_rate_threshold=0.55,     # 如果候选模型胜率达到 55% 则接受

        model_dir="models",          # 保存模型的目录
        save_every=1,                # 每次迭代保存模型
        pretrained_model_path=None,  # 预训练模型路径（None 表示从头训练）

        next_iteration_continuation=1  # 从第 1 次迭代开始
    )

