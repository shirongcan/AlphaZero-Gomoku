import os
import time
import random
from collections import deque
from typing import List, Tuple, Optional
import numpy as np
from network import PyTorchModel
from mcts.new_mcts_alpha import MCTS
from games.gomoku import Gomoku as GameClass
from datetime import datetime
from copy import deepcopy
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


# -------------------------
#  多进程自对弈 worker
# -------------------------
_SELFPLAY_MODEL = None
_SELFPLAY_MODEL_META = None  # (model_path, board_size, action_size, device)

# 评估 worker 模型缓存（每个子进程只加载一次）
_EVAL_MODEL_NEW = None
_EVAL_MODEL_BEST = None
_EVAL_MODEL_META = None  # (new_path, best_path, board_size, action_size, device)


def _selfplay_worker_init(
    model_path: str,
    board_size: int,
    action_size: int,
    device: str,
    base_seed: int = 0,
    torch_num_threads: int = 1,
):
    """
    Windows 下使用 spawn：每个子进程会重新 import 本文件。
    initializer 用于设定随机种子与 PyTorch 线程数，避免 CPU 过度抢占。
    """
    try:
        import torch
        torch.set_num_threads(max(1, int(torch_num_threads)))
    except Exception:
        pass

    pid = os.getpid()
    seed = int(base_seed) + pid
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    # 每个子进程只加载一次模型（避免每个任务重复 load）
    global _SELFPLAY_MODEL, _SELFPLAY_MODEL_META
    _SELFPLAY_MODEL_META = (str(model_path), int(board_size), int(action_size), str(device))

    from network import PyTorchModel  # 延迟 import
    _SELFPLAY_MODEL = PyTorchModel(board_size=int(board_size), action_size=int(action_size), device=str(device))
    _SELFPLAY_MODEL.load(str(model_path), map_location=str(device))


def _selfplay_generate_games(
    *,
    board_size: int,
    n_simulations: int,
    cpuct: float,
    temp_threshold: int,
    add_dirichlet_noise: bool,
    games_to_play: int,
    use_symmetries: bool,
    max_moves: int,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    dirichlet_n_moves: int,
    device: str = "cpu",
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], dict]:
    """
    子进程入口：加载模型 -> 跑若干局自对弈 -> 返回样本与胜负统计。
    注意：为了避免 GPU 多进程争用，默认 device=cpu。
    """
    from mcts.new_mcts_alpha import MCTS
    from games.gomoku import Gomoku as GameClass

    # 多进程路径：优先复用 initializer 加载的全局模型
    global _SELFPLAY_MODEL, _SELFPLAY_MODEL_META
    model = _SELFPLAY_MODEL
    if model is None:
        # 兼容：如果没有走 initializer（例如单进程/直接调用），再本地创建模型
        from network import PyTorchModel
        model = PyTorchModel(board_size=board_size, action_size=board_size * board_size, device=device)

    def temp_fn(move_number: int):
        return max(0.0, 1.0 - move_number / temp_threshold)

    winners = {0: 0, 1: 0, 2: 0}
    all_examples: List[Tuple[np.ndarray, np.ndarray, float]] = []

    for _ in range(int(games_to_play)):
        mcts_play = MCTS(
            game_class=GameClass,
            n_simulations=n_simulations,
            nn_model=model,
            cpuct=cpuct,
            dirichlet_alpha=dirichlet_alpha,
            epsilon=dirichlet_epsilon,
            apply_dirichlet_n_first_moves=dirichlet_n_moves,
            add_dirichlet_noise=add_dirichlet_noise,
        )
        game = GameClass(size=board_size)
        game.current_player = 1

        examples, winner = play_game_and_collect(
            mcts_play,
            game,
            temp_fn,
            max_moves=max_moves,
            use_symmetries=use_symmetries,
        )
        all_examples.extend(examples)
        winners[winner] = winners.get(winner, 0) + 1

        mcts_play.clear_tree()
        del mcts_play

    # 显式释放
    # 如果是全局复用模型，不在这里释放；让子进程退出时统一回收
    gc.collect()

    return all_examples, winners

# -------------------------
#  多进程评估 worker
# -------------------------
def _eval_worker_init(
    model_new_path: str,
    model_best_path: str,
    board_size: int,
    action_size: int,
    device: str,
    base_seed: int = 0,
    torch_num_threads: int = 1,
):
    try:
        import torch
        torch.set_num_threads(max(1, int(torch_num_threads)))
    except Exception:
        pass

    pid = os.getpid()
    seed = int(base_seed) + pid
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    global _EVAL_MODEL_NEW, _EVAL_MODEL_BEST, _EVAL_MODEL_META
    _EVAL_MODEL_META = (str(model_new_path), str(model_best_path), int(board_size), int(action_size), str(device))

    from network import PyTorchModel  # 延迟 import
    _EVAL_MODEL_NEW = PyTorchModel(board_size=int(board_size), action_size=int(action_size), device=str(device))
    _EVAL_MODEL_NEW.load(str(model_new_path), map_location=str(device))

    _EVAL_MODEL_BEST = PyTorchModel(board_size=int(board_size), action_size=int(action_size), device=str(device))
    _EVAL_MODEL_BEST.load(str(model_best_path), map_location=str(device))


def _eval_play_games(
    *,
    board_size: int,
    n_games: int,
    start_index: int,
    n_simulations: int,
    cpuct: float,
) -> Tuple[int, int, int]:
    """
    子进程评估：跑 n_games 局，使用 start_index 来决定交替先手。
    返回 (new_wins, draws, total_games)
    """
    from mcts.new_mcts_alpha import MCTS
    from games.gomoku import Gomoku as GameClass

    global _EVAL_MODEL_NEW, _EVAL_MODEL_BEST
    model_new = _EVAL_MODEL_NEW
    model_best = _EVAL_MODEL_BEST

    new_wins = 0
    draws = 0

    for gi in range(int(n_games)):
        global_i = int(start_index) + gi

        game = GameClass(size=int(board_size))
        # 随机第一手，增加开局多样性（评估时使用确定性选择，不随机会导致所有对局相同）
        # 第一手（玩家1）
        r1 = random.randint(0, int(board_size) - 3
        c1 = random.randint(0, int(board_size) - 3)
        game.do_move((r1, c1))
        # 现在 current_player = 2，从第二手开始真正评估

        new_starts = (global_i % 2 == 0)
        move_number = 1

        mcts_new = MCTS(
            game_class=GameClass,
            n_simulations=n_simulations,
            nn_model=model_new,
            cpuct=cpuct,
            add_dirichlet_noise=False,
        )
        mcts_best = MCTS(
            game_class=GameClass,
            n_simulations=n_simulations,
            nn_model=model_best,
            cpuct=cpuct,
            add_dirichlet_noise=False,
        )

        while not game.is_game_over():
            if (game.current_player == 1 and new_starts) or (game.current_player == 2 and not new_starts):
                pi = mcts_new.run(game, len(game.move_history))
            else:
                pi = mcts_best.run(game, len(game.move_history))

            action = int(np.argmax(pi))
            rr, cc = divmod(action, game.size)
            game.do_move((rr, cc))
            move_number += 1
            if move_number > game.size * game.size:
                break

        winner = game.get_winner()
        if winner == 0:
            draws += 1
        else:
            if (winner == 1 and new_starts) or (winner == 2 and not new_starts):
                new_wins += 1

        mcts_new.clear_tree()
        mcts_best.clear_tree()
        del mcts_new
        del mcts_best

    gc.collect()
    return int(new_wins), int(draws), int(n_games)

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
        pi = mcts.run(game, len(game.move_history))  # 向量 (action_size,) 第二个参数是当前是第几步
        # 这个参数是让MCtS 知道当前是第几步,是不是要加入dirichlet noise，用来增强MCTS的探索能力
        # 返回一个向量 (action_size,) 每个元素是每个动作的概率
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
                cpuct: float = 1.0) -> Tuple[int, float, int]:
    """
    在 model_new 和 model_best 之间进行 n_games 局游戏（轮流先手）。
    返回 (win_rate_of_new, draws)
    """
    # 仅支持 Gomoku（保留 game_name 参数用于向后兼容）
    rules_name = "gomoku"

    new_wins = 0
    draws = 0
    total = n_games

    for i in range(n_games):
        game = GameClass(size=model_new.board_size)

        # 随机第一手，增加开局多样性（评估时使用确定性选择，不随机会导致所有对局相同）
        # 第一手（玩家1）
        r1 = random.randint(0, model_new.board_size - 1)
        c1 = random.randint(0, model_new.board_size - 1)
        game.do_move((r1, c1))
        # 现在 current_player = 2，从第二手开始真正评估

        # 确定谁先手：新模型在偶数局先手
        new_starts = (i % 2 == 0)
        move_number = 1

        # 为两个玩家创建 MCTS 实例（扩展时各自使用自己的模型）
        mcts_new = MCTS(game_class=GameClass, n_simulations=n_simulations, nn_model=model_new, cpuct=cpuct, add_dirichlet_noise=False)
        mcts_best = MCTS(game_class=GameClass, n_simulations=n_simulations, nn_model=model_best, cpuct=cpuct, add_dirichlet_noise=False)

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
#  多进程评估（外部接口）
# -------------------------
def evaluate_models_mp(
    model_new: PyTorchModel,
    model_best: PyTorchModel,
    board_size: int,
    action_size: int,
    n_games: int,
    n_simulations: int,
    cpuct: float,
    *,
    model_dir: str,
    num_workers: int,
    games_per_task: int = 1,
    device: str = "cpu",
    base_seed: int = 54321,
    torch_threads: int = 1,
) -> Tuple[int, float, int]:
    """
    并行评估：保存两个模型 checkpoint -> 多进程并行跑对局 -> 汇总。
    返回 (new_wins, win_rate, draws)
    """
    os.makedirs(model_dir, exist_ok=True)

    # 保存 checkpoint（避免跨进程传递模型对象）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_new = os.path.join(model_dir, f"_eval_model_new_{ts}.pt")
    ckpt_best = os.path.join(model_dir, f"_eval_model_best_{ts}.pt")
    model_new.save(ckpt_new)
    model_best.save(ckpt_best)

    games_per_task = max(1, int(games_per_task))
    tasks = []
    remaining = int(n_games)
    start_idx = 0
    while remaining > 0:
        g = min(games_per_task, remaining)
        tasks.append((start_idx, g))
        start_idx += g
        remaining -= g

    ctx = mp.get_context("spawn")
    total_new_wins = 0
    total_draws = 0
    total_games = 0

    with ProcessPoolExecutor(
        max_workers=int(num_workers),
        mp_context=ctx,
        initializer=_eval_worker_init,
        initargs=(ckpt_new, ckpt_best, board_size, action_size, device, base_seed, torch_threads),
    ) as ex:
        futures = []
        for sidx, gcount in tasks:
            futures.append(
                ex.submit(
                    _eval_play_games,
                    board_size=board_size,
                    n_games=int(gcount),
                    start_index=int(sidx),
                    n_simulations=n_simulations,
                    cpuct=cpuct,
                )
            )

        for fut in as_completed(futures):
            nw, dr, tg = fut.result()
            total_new_wins += int(nw)
            total_draws += int(dr)
            total_games += int(tg)

    # 清理 checkpoint
    for p in (ckpt_new, ckpt_best):
        try:
            os.remove(p)
        except Exception:
            pass

    win_rate = total_new_wins / float(total_games) if total_games > 0 else 0.0
    return int(total_new_wins), float(win_rate), int(total_draws)


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
    # --- MCTS Dirichlet噪声参数 ---
    dirichlet_alpha: float = 0.03,             # Dirichlet噪声的alpha参数
    dirichlet_epsilon: float = 0.25,           # Dirichlet噪声的混合比例
    dirichlet_n_moves: int = 30,               # 前N手添加Dirichlet噪声
    # --- 多进程自对弈参数 ---
    selfplay_num_workers: int = 0,             # 0=自动（建议 CPU 核心数-1，最多8）
    selfplay_device: str = "cpu",              # 建议 "cpu"，避免 CUDA 多进程争用
    selfplay_games_per_task: int = 1,          # 每个任务包含的自对弈局数（越大 IPC 越少）
    selfplay_base_seed: int = 12345,           # 子进程随机种子基数
    selfplay_torch_threads: int = 1,           # 每个子进程内 torch CPU 线程数
    # --- 多进程评估参数 ---
    eval_num_workers: int = 0,                 # 0=自动（建议 CPU 核心数-1，最多8）
    eval_device: str = "cpu",                  # 建议 cpu（多进程 cuda 风险高）
    eval_games_per_task: int = 1,              # 每个任务评估局数
    eval_base_seed: int = 54321,
    eval_torch_threads: int = 1,
):
    """
    核心训练流程。
    """
    os.makedirs(model_dir, exist_ok=True)

    # 根据 board_size 计算动作空间大小
    action_size = board_size * board_size  # 对于 Gomoku，动作是棋盘上的位置

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
        # 关键：让候选模型复制最佳模型的初始权重，确保第一轮评估公平
        model_candidate.net.load_state_dict(model_best.net.state_dict())

    # 经验回放缓冲区
    buffer = ReplayBuffer(capacity=buffer_size)

    # 温度调度
    def temp_fn(move_number: int):
        return max(0.0, 1.0 - move_number / temp_threshold)

    for it in range(next_iteration_continuation, next_iteration_continuation + num_iterations):
        t0 = time.time()
        print(f"\n=== ITER {it}/{next_iteration_continuation + num_iterations - 1}: 自对弈生成 (games={games_per_iteration}, sims={n_simulations}) ===")
        selfplay_t0 = time.time()

        # 使用候选模型进行自对弈生成（多进程）
        winners = {0: 0, 1: 0, 2: 0}

        # 自动 worker 数：CPU 核心数-1，最多 8，最少 1
        if selfplay_num_workers and selfplay_num_workers > 0:
            num_workers = int(selfplay_num_workers)
        else:
            cpu_cnt = os.cpu_count() or 2
            num_workers = max(1, min(8, cpu_cnt - 1))

        # 1 worker 等价串行（但仍走同一套代码路径）
        max_moves = board_size * board_size
        use_symmetries = True
        add_dirichlet_noise = True

        # 并行执行
        if num_workers == 1:
            # 串行：直接使用主进程的 model_candidate（避免多余的 save/load）
            for g in range(games_per_iteration):
                mcts_play = MCTS(
                    game_class=GameClass,
                    n_simulations=n_simulations,
                    nn_model=model_candidate,
                    cpuct=cpuct,
                    dirichlet_alpha=dirichlet_alpha,
                    epsilon=dirichlet_epsilon,
                    apply_dirichlet_n_first_moves=dirichlet_n_moves,
                    add_dirichlet_noise=add_dirichlet_noise,
                )
                game = GameClass(size=board_size)
                game.current_player = 1
                examples, winner = play_game_and_collect(
                    mcts_play, game, temp_fn, max_moves=max_moves, use_symmetries=use_symmetries
                )
                buffer.add(examples)
                winners[winner] = winners.get(winner, 0) + 1

                mcts_play.clear_tree()
                del mcts_play
                gc.collect()
        else:
            # 保存候选模型到 checkpoint，子进程从磁盘加载（避免传递不可 pickle 的模型对象）
            selfplay_ckpt_path = os.path.join(model_dir, f"_selfplay_candidate_iter{it}.pt")
            model_candidate.save(selfplay_ckpt_path)

            # 任务切分
            games_per_task = max(1, int(selfplay_games_per_task))
            tasks = []
            remaining = int(games_per_iteration)
            while remaining > 0:
                g = min(games_per_task, remaining)
                tasks.append(g)
                remaining -= g

            gen_games_done = 0
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=_selfplay_worker_init,
                initargs=(selfplay_ckpt_path, board_size, action_size, selfplay_device, selfplay_base_seed, selfplay_torch_threads),
            ) as ex:
                futures = []
                for gcount in tasks:
                    futures.append(
                        ex.submit(
                            _selfplay_generate_games,
                            board_size=board_size,
                            n_simulations=n_simulations,
                            cpuct=cpuct,
                            temp_threshold=temp_threshold,
                            add_dirichlet_noise=add_dirichlet_noise,
                            games_to_play=int(gcount),
                            use_symmetries=use_symmetries,
                            max_moves=max_moves,
                            dirichlet_alpha=dirichlet_alpha,
                            dirichlet_epsilon=dirichlet_epsilon,
                            dirichlet_n_moves=dirichlet_n_moves,
                            device=selfplay_device,
                        )
                    )

                for fut in as_completed(futures):
                    examples, w = fut.result()
                    buffer.add(examples)
                    for k, v in w.items():
                        winners[k] = winners.get(k, 0) + int(v)
                    gen_games_done += int(sum(w.values()))

            # 可选：清理 selfplay checkpoint（保留也可以方便复现实验）
            try:
                os.remove(selfplay_ckpt_path)
            except Exception:
                pass

        selfplay_t1 = time.time()
        print(f"自对弈完成：耗时 {(selfplay_t1 - selfplay_t0):.1f}s，胜负统计={winners}，buffer_size={len(buffer)}")

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

        # 评估（精简输出：只在结束后汇总一次）
        eval_t0 = time.time()
        try:
            # 自动评估进程数（与自对弈同策略）
            if eval_num_workers and eval_num_workers > 0:
                eval_workers = int(eval_num_workers)
            else:
                cpu_cnt = os.cpu_count() or 2
                eval_workers = max(1, min(8, cpu_cnt - 1))

            if eval_workers == 1:
                new_wins, win_rate, draws = evaluate_models(
                    model_candidate,
                    model_best,
                    game_name,
                    n_games=eval_games,
                    n_simulations=eval_mcts_simulations,
                    cpuct=cpuct,
                )
            else:
                new_wins, win_rate, draws = evaluate_models_mp(
                    model_candidate,
                    model_best,
                    board_size=board_size,
                    action_size=action_size,
                    n_games=eval_games,
                    n_simulations=eval_mcts_simulations,
                    cpuct=cpuct,
                    model_dir=model_dir,
                    num_workers=eval_workers,
                    games_per_task=eval_games_per_task,
                    device=eval_device,
                    base_seed=eval_base_seed,
                    torch_threads=eval_torch_threads,
                )
        except Exception as e:
            # 保持可见性，但不刷屏
            print(f"评估失败：{e}")
            new_wins, win_rate, draws = 0, 0.0, 0

        eval_t1 = time.time()
        print(
            f"评估完成：耗时 {(eval_t1 - eval_t0):.1f}s，胜率={win_rate:.3f}（{new_wins}/{eval_games}），平局={draws}"
        )

        # 接受/拒绝
        if win_rate >= win_rate_threshold:
            print(" 候选模型被接受 -> 提升为最佳模型。")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(model_dir, f"model_best_iter{it}_{timestamp}.pt")
            model_candidate.save(path)
            # 更新 model_best（深拷贝权重和优化器状态）
            model_best.net.load_state_dict(model_candidate.net.state_dict())
            model_best.optimizer.load_state_dict(model_candidate.optimizer.state_dict())
            # 从最佳模型创建新的候选模型
            model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
            model_candidate.net.load_state_dict(model_best.net.state_dict())
            model_candidate.optimizer.load_state_dict(model_best.optimizer.state_dict())
        else:
            print(" 候选模型被拒绝 -> 从最佳模型恢复候选模型（重置优化器）。")
            # 从最佳模型权重重置候选模型，但不继承优化器状态（给予新的开始）
            model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
            model_candidate.net.load_state_dict(model_best.net.state_dict())
            # 不加载优化器状态，让优化器保持初始化状态，避免陷入局部最优

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
    mp.freeze_support()
    train_alphazero(
        game_name="gomoku",           # 游戏 Gomoku
        board_size=15,                # 棋盘大小 (15x15)

        num_iterations=30,           # 30 次训练迭代
        games_per_iteration=70,       # 每次迭代 70 局游戏

        n_simulations=1600,          # MCTS 1600 次模拟
        cpuct=1.0,                   # MCTS 的探索/利用平衡因子

        buffer_size=60000,           # 经验回放缓冲区，最多 60,000 个样本
        batch_size=128,               # 每个训练批次 128 个样本
        epochs_per_iter=3,           # 每次迭代 3 个训练轮次

        temp_threshold=10,           # 探索温度阈值
        eval_games=50,               # 50 局评估游戏（提高统计稳定性）
        eval_mcts_simulations=1600,  # 评估时 MCTS 1600 次模拟
        win_rate_threshold=0.52,     # 如果候选模型胜率达到 52% 则接受

        # Dirichlet噪声参数（AlphaZero标准配置）
        dirichlet_alpha=0.03,        # Dirichlet噪声的alpha参数（围棋论文标准值）
        dirichlet_epsilon=0.25,      # 噪声混合比例（根节点探索）
        dirichlet_n_moves=10,        # 前30手添加噪声（增加开局多样性）

        model_dir="models",          # 保存模型的目录
        save_every=1,                # 每次迭代保存模型
        pretrained_model_path="models/snapshot_iter100_20260109_025843.pt",  # 预训练模型路径（None 表示从头训练）

        next_iteration_continuation=101,  # 从第 101 次迭代开始

        # 多进程自对弈：28 个进程
        selfplay_num_workers=28,
        selfplay_device="cpu",
        selfplay_games_per_task=1,
        selfplay_torch_threads=1,

        # 多进程评估：28 个进程
        eval_num_workers=28,
        eval_device="cpu",
        eval_games_per_task=1,
        eval_torch_threads=1,
    )