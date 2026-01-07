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
from datetime import datetime
from copy import deepcopy
import sys
import gc
import multiprocessing as mp
from functools import partial
import tempfile
import pickle
import signal
import re
import logging

# 在循环内动态映射游戏名称到类

# -------------------------
#  日志系统
# -------------------------
class TeeOutput:
    """同时输出到控制台和文件"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 立即刷新到文件
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.close()


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


def determine_n_workers(n_workers: Optional[int], n_tasks: int, prefix: str = "") -> int:
    """
    确定工作进程数。如果 n_workers 为 None，则自动计算（备用机制）。
    通常在程序启动阶段已经确定了工作进程数，这里只在需要时作为备用。
    
    Args:
        n_workers: 指定的工作进程数，None 表示自动计算
        n_tasks: 任务数量（用于限制工作进程数不超过任务数）
        prefix: 打印信息的前缀（用于区分不同阶段）
    
    Returns:
        确定的工作进程数（不超过任务数）
    """
    if n_workers is None:
        # 备用机制：如果启动阶段未确定，则自动计算
        total_cores = mp.cpu_count()
        available_cores = max(1, total_cores - 4)
        n_workers = min(available_cores, n_tasks)
        prefix_str = f"{prefix} - " if prefix else ""
        print(f"  {prefix_str}CPU核心数: {total_cores}，可用核心数: {available_cores}，使用工作进程数: {n_workers}")
    else:
        # 如果已经指定，确保不超过任务数
        n_workers = min(n_workers, n_tasks)
    return n_workers


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
    
    def save(self, filepath: str):
        """保存缓冲区到文件"""
        with open(filepath, 'wb') as f:
            # 将deque转换为list以便序列化
            buffer_list = list(self.buffer)
            pickle.dump(buffer_list, f)
    
    def load(self, filepath: str):
        """从文件加载缓冲区"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                buffer_list = pickle.load(f)
                # 清空当前缓冲区并加载数据
                self.buffer.clear()
                for ex in buffer_list:
                    self.buffer.append(ex)


# -------------------------
#  自对弈单局游戏（单进程版本）
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
        
        # 确保策略只包含合法动作并归一化
        valid_mask = game.get_valid_moves()
        pi = pi * valid_mask
        pi_sum = np.sum(pi)
        if pi_sum > 1e-8:
            pi = pi / pi_sum
        else:
            # 如果所有动作都被掩码（不应该发生），使用均匀分布
            pi = valid_mask / (np.sum(valid_mask) + 1e-8)
        
        pi_for_store = pi.copy()

        temp = temp_fn(move_number)
        action = sample_action_from_pi(pi, temp)

        # 安全回退：如果选择的动作不合法，使用 argmax
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
#  多进程工作函数：在子进程中执行单局游戏
# -------------------------
def _worker_play_game(args):
    """
    工作函数：在子进程中执行一局游戏
    args: (model_state_dict_path, game_name, board_size, n_simulations, cpuct, 
           temp_threshold, max_moves, use_symmetries, seed)
    """
    (model_state_dict_path, game_name, board_size, n_simulations, cpuct,
     temp_threshold, max_moves, use_symmetries, seed) = args
    
    # 设置随机种子（每个进程不同）
    np.random.seed(seed)
    random.seed(seed)
    
    # 导入必要的模块（在子进程中）
    from games.gomoku import Gomoku
    from network import PyTorchModel
    from mcts.new_mcts_alpha import MCTS
    
    # 导入工具函数（在子进程中）
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
    
    # 只使用Gomoku（仅进行Gomoku训练）
    GameClass = Gomoku
    
    # 创建模型并加载权重
    action_size = board_size * board_size
    model = PyTorchModel(board_size=board_size, action_size=action_size)
    model.load(model_state_dict_path)
    
    # 创建 MCTS
    mcts = MCTS(
        game_class=GameClass,
        n_simulations=n_simulations,
        nn_model=model,
        cpuct=cpuct,
        dirichlet_alpha=0.03,
        epsilon=0.03,
        apply_dirichlet_n_first_moves=10,
        add_dirichlet_noise=True
    )
    
    # 创建游戏
    game = GameClass(size=board_size)
    game.current_player = 1
    
    # 温度函数
    def temp_fn(move_number: int):
        return max(0.0, 1.0 - move_number / temp_threshold)
    
    # 执行游戏
    examples = []
    move_number = 0
    
    while True:
        state_enc = game.get_encoded_state()
        pi = mcts.run(game, len(game.move_history))
        
        # 确保策略只包含合法动作并归一化
        valid_mask = game.get_valid_moves()
        pi = pi * valid_mask
        pi_sum = np.sum(pi)
        if pi_sum > 1e-8:
            pi = pi / pi_sum
        else:
            # 如果所有动作都被掩码（不应该发生），使用均匀分布
            pi = valid_mask / (np.sum(valid_mask) + 1e-8)
        
        pi_for_store = pi.copy()
        
        temp = temp_fn(move_number)
        action = sample_action_from_pi(pi, temp)
        
        # 安全回退
        if valid_mask[action] != 1.0:
            action = int(np.argmax(pi))
        
        examples.append((state_enc, pi_for_store, int(game.current_player)))
        
        r, c = divmod(action, game.size)
        game.do_move((r, c))
        move_number += 1
        
        if game.is_game_over() or move_number >= max_moves:
            break
    
    winner = game.get_winner()
    
    # 处理对称性和奖励
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
    
    # 清理
    mcts.clear_tree()
    del mcts, model, game
    gc.collect()
    
    return final_examples, winner


# -------------------------
#  并行自对弈生成（多进程版本）
# -------------------------
def play_games_parallel(model_candidate: PyTorchModel,
                       game_name: str,
                       board_size: int,
                       n_games: int,
                       n_simulations: int,
                       cpuct: float,
                       temp_threshold: int,
                       max_moves: int,
                       use_symmetries: bool,
                       n_workers: Optional[int] = None) -> Tuple[List, dict]:
    """
    使用多进程并行生成多局游戏
    
    返回:
        all_examples: 所有游戏的示例列表
        winners: {0: count, 1: count, 2: count}
    """
    # 设置信号处理（仅在非 Windows 系统上，Windows 使用不同的机制）
    if sys.platform != "win32":
        def signal_handler(signum, frame):
            raise KeyboardInterrupt("收到中断信号")
        signal.signal(signal.SIGINT, signal_handler)
    
    n_workers = determine_n_workers(n_workers, n_games, "")
    
    # 保存模型权重到临时文件（用于子进程加载）
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as f:
        temp_model_path = f.name
        model_candidate.save(temp_model_path)
    
    try:
        # 准备参数
        seeds = [random.randint(0, 2**31 - 1) for _ in range(n_games)]
        args_list = [
            (temp_model_path, game_name, board_size, n_simulations, cpuct,
             temp_threshold, max_moves, use_symmetries, seed)
            for seed in seeds
        ]
        
        # 记录开始时间
        start_time = time.time()
        start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  开始时间: {start_time_str}")
        print(f"  使用 {n_workers} 个进程并行生成 {n_games} 局游戏...")
        print(f"  提示: 按 CTRL-C 可以中断训练")
        
        # 使用进程池并行执行（使用异步方法以支持中断）
        pool = None
        results = None
        try:
            pool = mp.Pool(processes=n_workers)
            # 使用 map_async 替代 map，这样可以响应中断
            async_result = pool.map_async(_worker_play_game, args_list)
            
            # 使用轮询机制等待结果，这样可以响应 KeyboardInterrupt
            # 在 Windows 上，get(timeout=None) 可能无法响应中断
            while True:
                try:
                    # 使用短超时轮询，这样可以在每次循环中检查中断
                    results = async_result.get(timeout=1.0)
                    break  # 如果成功获取结果，退出循环
                except mp.TimeoutError:
                    # 超时是正常的，继续轮询
                    continue
                except KeyboardInterrupt:
                    print("\n  检测到中断信号 (CTRL-C)，正在终止进程池...")
                    # 立即终止所有子进程
                    if pool is not None:
                        try:
                            pool.terminate()
                        except:
                            pass
                    # 等待进程结束，捕获可能的 KeyboardInterrupt
                    if pool is not None:
                        try:
                            pool.join(timeout=5)
                        except KeyboardInterrupt:
                            # 如果 join 时也收到中断，继续清理
                            pass
                        except:
                            pass
                    print("  进程池已终止")
                    raise  # 重新抛出异常，让上层处理
        
        except KeyboardInterrupt:
            # 确保进程池被清理
            if pool is not None:
                try:
                    pool.terminate()
                except:
                    pass
                try:
                    pool.join(timeout=2)  # 缩短超时时间
                except (KeyboardInterrupt, Exception):
                    # 忽略所有异常，确保程序能继续退出
                    pass
            raise
        finally:
            # 确保进程池被正确关闭
            if pool is not None:
                try:
                    pool.close()
                except:
                    pass
                try:
                    pool.join(timeout=1)  # 缩短超时时间
                except (KeyboardInterrupt, Exception):
                    # 忽略所有异常
                    pass
        
        # 如果没有结果（被中断），返回空结果
        if results is None:
            return [], {0: 0, 1: 0, 2: 0}
        
        # 记录结束时间
        end_time = time.time()
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_time = end_time - start_time
        
        # 收集结果
        all_examples = []
        winners = {0: 0, 1: 0, 2: 0}
        
        for i, (examples, winner) in enumerate(results):
            all_examples.extend(examples)
            winners[winner] = winners.get(winner, 0) + 1
        
        # 所有游戏完成后统一输出汇总信息
        total_examples = len(all_examples)
        print(f"  结束时间: {end_time_str}")
        print(f"  耗时: {elapsed_time:.1f}秒 ({elapsed_time/60:.1f}分钟)")
        print(f"  并行生成完成: {n_games} 局游戏，共生成 {total_examples} 个训练样本")
        print(f"  获胜统计: 玩家1={winners.get(1, 0)}, 玩家2={winners.get(2, 0)}, 平局={winners.get(0, 0)}")
        
        return all_examples, winners
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_model_path):
            os.unlink(temp_model_path)


# -------------------------
#  多进程评估工作函数：在子进程中执行单局评估游戏
# -------------------------
def _worker_evaluate_game(args):
    """
    工作函数：在子进程中执行一局评估游戏
    args: (model_new_path, model_best_path, game_name, board_size, game_index,
           n_simulations, cpuct, seed)
    返回: (new_wins, draws) - 新模型是否获胜(1/0)，是否平局(1/0)
    """
    (model_new_path, model_best_path, game_name, board_size, game_index,
     n_simulations, cpuct, seed) = args
    
    # 设置随机种子（每个进程不同）
    np.random.seed(seed)
    random.seed(seed)
    
    # 导入必要的模块（在子进程中）
    from games.gomoku import Gomoku
    from network import PyTorchModel
    from mcts.new_mcts_alpha import MCTS
    
    # 只使用Gomoku（仅进行Gomoku训练）
    GameClass = Gomoku
    
    # 创建模型并加载权重
    action_size = board_size * board_size
    model_new = PyTorchModel(board_size=board_size, action_size=action_size)
    model_new.load(model_new_path)
    
    model_best = PyTorchModel(board_size=board_size, action_size=action_size)
    model_best.load(model_best_path)
    
    # 创建游戏
    game = GameClass(size=board_size)
    
    # 随机初始移动
    r = random.randint(0, board_size - 1)
    c = random.randint(0, board_size - 1)
    game.do_move((r, c))
    
    # 确定谁先手：新模型在偶数局先手
    new_starts = (game_index % 2 == 0)
    move_number = 1
    
    # 为两个玩家创建 MCTS 实例
    mcts_new = MCTS(
        game_class=GameClass,
        n_simulations=n_simulations,
        nn_model=model_new,
        cpuct=cpuct,
        dirichlet_alpha=0.03,
        epsilon=0.03,
        apply_dirichlet_n_first_moves=10,
        add_dirichlet_noise=False
    )
    mcts_best = MCTS(
        game_class=GameClass,
        n_simulations=n_simulations,
        nn_model=model_best,
        cpuct=cpuct,
        dirichlet_alpha=0.03,
        epsilon=0.03,
        apply_dirichlet_n_first_moves=10,
        add_dirichlet_noise=False
    )
    
    # 执行游戏
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
    
    # 确定新模型是否获胜
    new_wins = 0
    draws = 0
    if winner == 0:
        draws = 1
    else:
        # 确定新模型是否获胜
        if (winner == 1 and new_starts) or (winner == 2 and not new_starts):
            new_wins = 1
    
    # 清理
    mcts_new.clear_tree()
    mcts_best.clear_tree()
    del mcts_new, mcts_best, model_new, model_best, game
    gc.collect()
    
    return new_wins, draws


# -------------------------
#  模型间评估（多进程版本）
# -------------------------
def evaluate_models(model_new: PyTorchModel,
                model_best: PyTorchModel,
                game_name: str,
                n_games: int = 20,
                n_simulations: int = 100,
                cpuct: float = 1.0,
                n_workers: Optional[int] = None) -> Tuple[int, float, int]:
    """
    在 model_new 和 model_best 之间进行 n_games 局游戏（轮流先手）。
    使用多进程并行执行评估游戏。
    返回 (new_wins, win_rate, draws)
    """
    n_workers = determine_n_workers(n_workers, n_games, "评估阶段")
    
    # 保存模型权重到临时文件（用于子进程加载）
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as f:
        temp_model_new_path = f.name
        model_new.save(temp_model_new_path)
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as f:
        temp_model_best_path = f.name
        model_best.save(temp_model_best_path)
    
    try:
        # 准备参数
        seeds = [random.randint(0, 2**31 - 1) for _ in range(n_games)]
        args_list = [
            (temp_model_new_path, temp_model_best_path, game_name, model_new.board_size, i,
             n_simulations, cpuct, seed)
            for i, seed in enumerate(seeds)
        ]
        
        # 记录开始时间
        start_time = time.time()
        start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  评估阶段 - 开始时间: {start_time_str}")
        print(f"  评估阶段 - 使用 {n_workers} 个进程并行评估 {n_games} 局游戏...")
        
        # 使用进程池并行执行
        pool = None
        results = None
        try:
            pool = mp.Pool(processes=n_workers)
            # 使用 map_async 替代 map，这样可以响应中断
            async_result = pool.map_async(_worker_evaluate_game, args_list)
            
            # 使用轮询机制等待结果，这样可以响应 KeyboardInterrupt
            while True:
                try:
                    # 使用短超时轮询，这样可以在每次循环中检查中断
                    results = async_result.get(timeout=1.0)
                    break  # 如果成功获取结果，退出循环
                except mp.TimeoutError:
                    # 超时是正常的，继续轮询
                    continue
                except KeyboardInterrupt:
                    print("\n  评估阶段 - 检测到中断信号 (CTRL-C)，正在终止进程池...")
                    # 立即终止所有子进程
                    if pool is not None:
                        try:
                            pool.terminate()
                        except:
                            pass
                    # 等待进程结束
                    if pool is not None:
                        try:
                            pool.join(timeout=5)
                        except KeyboardInterrupt:
                            pass
                        except:
                            pass
                    print("  评估阶段 - 进程池已终止")
                    raise  # 重新抛出异常，让上层处理
        
        except KeyboardInterrupt:
            # 确保进程池被清理
            if pool is not None:
                try:
                    pool.terminate()
                except:
                    pass
                try:
                    pool.join(timeout=2)
                except (KeyboardInterrupt, Exception):
                    pass
            raise
        finally:
            # 确保进程池被正确关闭
            if pool is not None:
                try:
                    pool.close()
                except:
                    pass
                try:
                    pool.join(timeout=1)
                except (KeyboardInterrupt, Exception):
                    pass
        
        # 如果没有结果（被中断），返回默认值
        if results is None:
            return 0, 0.0, 0
        
        # 记录结束时间
        end_time = time.time()
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_time = end_time - start_time
        
        # 收集结果
        new_wins = 0
        draws = 0
        for wins, draw in results:
            new_wins += wins
            draws += draw
        
        total = n_games
        win_rate = new_wins / float(total) if total > 0 else 0.0
        
        print(f"  评估阶段 - 结束时间: {end_time_str}")
        print(f"  评估阶段 - 耗时: {elapsed_time:.1f}秒 ({elapsed_time/60:.1f}分钟)")
        
        return new_wins, win_rate, draws
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_model_new_path):
            os.unlink(temp_model_new_path)
        if os.path.exists(temp_model_best_path):
            os.unlink(temp_model_best_path)


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
    next_iteration_continuation: int = 1,
    n_workers: Optional[int] = None  # 多进程工作进程数，None 表示自动选择
):
    """
    核心训练流程。
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建训练数据保存目录
    training_data_dir = os.path.join(model_dir, "training_data")
    os.makedirs(training_data_dir, exist_ok=True)

    # 根据 board_size 计算动作空间大小
    action_size = board_size * board_size  # 对于 Gomoku，动作是棋盘上的位置

    # 检查是否存在预训练模型
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"从以下路径加载预训练模型: {pretrained_model_path}")
        model_best = PyTorchModel(board_size=board_size, action_size=action_size)
        model_best.load(pretrained_model_path)  # 加载预训练模型
        model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
        model_candidate.net.load_state_dict(model_best.net.state_dict())
        model_candidate.optimizer.load_state_dict(model_best.optimizer.state_dict())
        print("预训练模型加载成功。")
    else:
        print("未找到预训练模型。初始化新模型。")
        model_best = PyTorchModel(board_size=board_size, action_size=action_size)
        model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)

    # 经验回放缓冲区
    buffer = ReplayBuffer(capacity=buffer_size)
    
    # 加载之前保存的buffer（如果存在）- 只在程序启动时加载一次
    buffer_filepath = os.path.join(training_data_dir, "training_data_buffer.pkl")
    if os.path.exists(buffer_filepath):
        try:
            buffer.load(buffer_filepath)
            print(f"\n从 {buffer_filepath} 加载了 {len(buffer)} 个训练样本到缓冲区。\n")
        except Exception as e:
            print(f"\n警告: 加载训练数据失败: {e}，从空缓冲区开始。\n")
    else:
        print("未找到已保存的训练数据，从空缓冲区开始。\n")

    # 温度调度
    def temp_fn(move_number: int):
        return max(0.0, 1.0 - move_number / temp_threshold)

    try:
        for it in range(next_iteration_continuation, next_iteration_continuation + num_iterations + 1):
            t0 = time.time()
            print(f"\n=== ITER {it}/{next_iteration_continuation + num_iterations}: 自对弈生成 (games={games_per_iteration}, sims={n_simulations}) ===")

            # 使用候选模型进行自对弈生成（多进程并行版本）
            # 可以通过设置 use_multiprocessing=False 来使用单进程版本
            use_multiprocessing = True  # 设置为 False 以使用单进程版本（用于调试）
            
            if use_multiprocessing:
                all_examples, winners = play_games_parallel(
                    model_candidate=model_candidate,
                    game_name=game_name,
                    board_size=board_size,
                    n_games=games_per_iteration,
                    n_simulations=n_simulations,
                    cpuct=cpuct,
                    temp_threshold=temp_threshold,
                    max_moves=board_size * board_size,
                    use_symmetries=True,
                    n_workers=n_workers  # 使用传入的参数，None 表示自动选择（CPU核心数）
                )
                buffer.add(all_examples)
                print(f"  缓冲区大小: {len(buffer)}")
                
                # 保存整个buffer到硬盘（覆盖之前的保存）
                buffer_filepath = os.path.join(training_data_dir, "training_data_buffer.pkl")
                buffer.save(buffer_filepath)
                print(f"  缓冲区已保存到: {buffer_filepath} ({len(buffer)} 个样本)")
            else:
                # 单进程版本（原始代码，用于调试）
                # 记录开始时间
                start_time = time.time()
                start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"  开始时间: {start_time_str}")
                print(f"  单进程生成 {games_per_iteration} 局游戏...")
                
                winners = {0: 0, 1: 0, 2: 0}
                all_examples = []
                for g in range(games_per_iteration):
                    mcts_play = MCTS(game_class=GameClass, n_simulations=n_simulations, nn_model=model_candidate, cpuct=cpuct, dirichlet_alpha=0.03, epsilon=0.03, apply_dirichlet_n_first_moves=10, add_dirichlet_noise=True)
                    game = GameClass(size=board_size)
                    game.current_player = 1
                    examples, winner = play_game_and_collect(mcts_play, game, temp_fn, max_moves=board_size * board_size, use_symmetries=True)
                    all_examples.extend(examples)
                    winners[winner] = winners.get(winner, 0) + 1

                    mcts_play.clear_tree()
                    del mcts_play
                    gc.collect()
                
                # 记录结束时间
                end_time = time.time()
                end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elapsed_time = end_time - start_time
                
                # 所有游戏完成后统一输出
                buffer.add(all_examples)
                total_examples = len(all_examples)
                print(f"  结束时间: {end_time_str}")
                print(f"  耗时: {elapsed_time:.1f}秒 ({elapsed_time/60:.1f}分钟)")
                print(f"  单进程生成完成: {games_per_iteration} 局游戏，共生成 {total_examples} 个训练样本")
                print(f"  获胜统计: 玩家1={winners.get(1, 0)}, 玩家2={winners.get(2, 0)}, 平局={winners.get(0, 0)}")
                print(f"  缓冲区大小: {len(buffer)}")
                
                # 保存整个buffer到硬盘（覆盖之前的保存）
                buffer_filepath = os.path.join(training_data_dir, "training_data_buffer.pkl")
                buffer.save(buffer_filepath)
                print(f"  缓冲区已保存到: {buffer_filepath} ({len(buffer)} 个样本)")

            # 如果有足够的样本，训练候选模型
            if len(buffer) >= batch_size:
                print(f"\nTraining candidate model: buffer={len(buffer)}, batch_size={batch_size}, epochs_per_iter={epochs_per_iter}")
                n_batches = max(1, len(buffer) // batch_size)
                for epoch in range(epochs_per_iter):
                    epoch_t0 = time.time()
                    
                    # 累积所有 batch 的 loss
                    epoch_policy_loss = 0.0
                    epoch_value_loss = 0.0
                    epoch_total_loss = 0.0
                    
                    for b in range(n_batches):
                        states_b, pis_b, zs_b = buffer.sample(batch_size)
                        loss_info = model_candidate.train_batch(states_b, pis_b, zs_b, epochs=1)
                        
                        # 累积 loss
                        epoch_policy_loss += loss_info['policy_loss']
                        epoch_value_loss += loss_info['value_loss']
                        epoch_total_loss += loss_info['total_loss']
                    
                    # 计算并打印平均 loss
                    avg_loss = {
                        'policy_loss': epoch_policy_loss / n_batches,
                        'value_loss': epoch_value_loss / n_batches,
                        'total_loss': epoch_total_loss / n_batches
                    }
                    
                    epoch_t1 = time.time()
                    print(f"  epoch {epoch+1}/{epochs_per_iter} finished in {epoch_t1 - epoch_t0:.1f}s, avg_loss={avg_loss}")
            else:
                print(f"训练样本不足 (buffer={len(buffer)}, 需要 {batch_size})。跳过本次迭代的训练。")

            # 评估
            print("\nEvaluating candidate vs best...")
            try:
                new_wins, win_rate, draws = evaluate_models(model_candidate, model_best, game_name, n_games=eval_games, n_simulations=eval_mcts_simulations, cpuct=cpuct, n_workers=n_workers)
            except Exception as e:
                print("Evaluation failed (exception):", e)
                win_rate, draws = 0.0, 0
                new_wins = 0

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
            print(f"迭代 {it} 完成，耗时 {(t1 - t0)/60:.1f}分钟。本次迭代获胜者: {winners}")
    
    except KeyboardInterrupt:
        print("\n\n=== 训练被用户中断 (CTRL-C) ===")
        print("程序退出，不保留任何内容。")
        sys.exit(0)
    
    print("\n=== 训练完成 ===")

# -------------------------
#  入口点
# -------------------------
if __name__ == "__main__":
    # 设置日志文件：将控制台输出同时保存到文件
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(log_dir, f"train_{timestamp}.log")
    tee_output = TeeOutput(log_filepath)
    sys.stdout = tee_output
    sys.stderr = tee_output  # 同时也捕获错误输出
    
    print(f"训练日志文件: {log_filepath}\n")
    
    try:
        # Windows 兼容性：设置 multiprocessing 启动方法
        # 注意：在 Windows 上，spawn 方法会创建新的 Python 解释器，所以 CTRL-C 处理需要特殊处理
        if sys.platform == "win32":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                # 如果已经设置过，忽略错误
                pass

        # 找到models里面最新的模型
        import re

        model_dir = "models"
        # 查找所有 snapshot*.pt 文件
        snapshot_files = [f for f in os.listdir(model_dir) if f.startswith('snapshot_') and f.endswith('.pt')]
        if len(snapshot_files) == 0:
            print("没有找到任何快照模型 (snapshot_*.pt)。")
            latest_snapshot = None
            next_iter = 1
        else:
            # 以迭代号排序，选出最新
            def extract_iter_num(filename):
                match = re.search(r"snapshot_iter(\d+)_", filename)
                return int(match.group(1)) if match else -1

            snapshot_files.sort(key=lambda x: (extract_iter_num(x), os.path.getmtime(os.path.join(model_dir, x))))
            latest_snapshot = snapshot_files[-1]
            print(f"最新快照模型: {latest_snapshot}")

            # 解析下一个迭代号
            latest_iter = extract_iter_num(latest_snapshot)
            next_iter = latest_iter + 1
            print(f"下一个迭代号: {next_iter}")

        #  pretrained_model_path 是最新快照模型路径
        pretrained_model_path = os.path.join(model_dir, latest_snapshot) if latest_snapshot else None
       
        # 在程序启动阶段确定工作进程数
        n_workers_user = None  # 用户指定的工作进程数，None 表示自动选择
        if n_workers_user is None:
            total_cores = mp.cpu_count()
            available_cores = max(1, total_cores - 4)  # 至少保留4个核心给其他应用
            n_workers = available_cores
            print(f"\n=== 系统信息 ===")
            print(f"CPU总核心数: {total_cores}")
            print(f"可用核心数: {available_cores} (已保留4个核心)")
            print(f"工作进程数: {n_workers}")
            print(f"===============\n")
        else:
            n_workers = n_workers_user
            print(f"\n使用指定的工作进程数: {n_workers}\n")

        train_alphazero(
            game_name="gomoku",           # 游戏 Gomoku
            board_size=15,                # 棋盘大小 (15x15)

            num_iterations=100,           # 30 次训练迭代
            games_per_iteration=60,       # 每次迭代 60 局游戏

            n_simulations=2000,          # MCTS 2000 次模拟
            cpuct=1.0,                   # MCTS 的探索/利用平衡因子

            buffer_size=60000,           # 经验回放缓冲区，最多 60,000 个样本（容纳约3次迭代）
            batch_size=128,               # 每个训练批次 128 个样本
            epochs_per_iter=5,           # 每次迭代 3 个训练轮次（初期数据质量低，少训练避免过拟合；后期可增加）

            temp_threshold=10,           # 探索温度阈值
            eval_games=40,               # 30 局评估游戏（提高统计稳定性）
            eval_mcts_simulations=2000,  # 评估时 MCTS 2000 次模拟
            win_rate_threshold=0.52,     # 如果候选模型胜率达到 50.3% 则接受

            model_dir="models",          # 保存模型的目录
            save_every=1,                # 每次迭代保存模型
            pretrained_model_path=pretrained_model_path,  # 预训练模型路径（None 表示从头训练）

            next_iteration_continuation=next_iter,  # 从第 1 次迭代开始
            n_workers=n_workers           # 工作进程数（已在启动阶段确定）
        )
    
    finally:
        # 恢复标准输出并关闭日志文件
        if 'tee_output' in locals():
            sys.stdout = tee_output.terminal
            sys.stderr = sys.__stderr__
            tee_output.close()
            print(f"\n日志已保存到: {log_filepath}")

