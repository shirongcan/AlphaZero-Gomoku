import numpy as np


def infer_current_player(board, board_array: np.ndarray) -> int:
    """
    优先从传入的 board/game 对象读取 current_player；
    若 board 只是 list/ndarray，则用棋盘已落子数推断（默认 1 先手）。
    """
    if hasattr(board, "current_player"):
        return int(getattr(board, "current_player"))

    stones = int(np.count_nonzero(board_array))
    return 1 if stones % 2 == 0 else 2


def infer_move_number(board, board_array: np.ndarray) -> int:
    """
    返回“已下了多少手”（用于 MCTS 的 move_number / 温度/噪声调度）。
    """
    if hasattr(board, "move_history"):
        try:
            return int(len(getattr(board, "move_history")))
        except Exception:
            pass
    return int(np.count_nonzero(board_array))


def infer_last_move(board, last_move_fallback):
    """
    优先从 board/game 对象读取 last_move；否则回退到调用方传入的 last_move。
    """
    if hasattr(board, "last_move"):
        return getattr(board, "last_move")
    return last_move_fallback


