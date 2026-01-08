import sys
import os
from typing import Optional, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from mcts.new_mcts_alpha import MCTS
from network import PyTorchModel
from games.gomoku import Gomoku
from players.utils import infer_current_player, infer_last_move, infer_move_number


def _list_model_files(models_dir: str = "models") -> List[str]:
    try:
        names = [
            n
            for n in os.listdir(models_dir)
            if n.lower().endswith(".pt") and os.path.isfile(os.path.join(models_dir, n))
        ]
    except FileNotFoundError:
        return []

    names.sort(key=lambda n: os.path.getmtime(os.path.join(models_dir, n)), reverse=True)
    return [os.path.join(models_dir, n) for n in names]


class Player:
    """
    AlphaZero（仅 Gomoku）玩家：
    - 不修改原 `players/player_alpha.py`
    - 默认从 models/ 里选择“最新修改时间”的 .pt 作为模型
    - GUI 可显式传入 model_path
    """

    def __init__(
        self,
        rules: str = "gomoku",
        board_size: int = 15,
        n_simulations: int = 3000,
        c_puct: float = 1.0,
        model_path: Optional[str] = None,
        nn_model=PyTorchModel,
    ):
        self.rules = (rules or "gomoku").lower()
        self.board_size = board_size
        self.n_simulations = n_simulations
        self.c_puct = c_puct

        if self.rules != "gomoku":
            # 明确只支持 gomoku，避免误用导致隐性行为
            raise ValueError("player_alpha_gomoku 仅支持 rules='gomoku'")

        if model_path is None:
            candidates = _list_model_files("models")
            model_path = candidates[0] if candidates else None

        self.model_path = model_path

        # 1) 创建网络
        self.net = nn_model(board_size=self.board_size)

        # 2) 加载模型（如果有）
        if self.model_path is not None:
            print(f"[PlayerAlphaGomoku] 加载模型: {self.model_path}")
            self.net.load(self.model_path)
        else:
            print("[PlayerAlphaGomoku] 警告：未提供模型且 models/ 为空，将使用随机权重。")

        # 3) eval 模式
        self.net.net.eval()

        # 4) MCTS
        self.game_class = Gomoku
        self.mcts = MCTS(
            game_class=self.game_class,
            n_simulations=self.n_simulations,
            nn_model=self.net,
            cpuct=self.c_puct,
            add_dirichlet_noise=False,
        )

    def play(self, board, turn_number, last_opponent_move):
        game = self.game_class(size=self.board_size)

        if isinstance(board, list):
            game.board = np.array(board, dtype=int)
        else:
            game.board = np.copy(board.board)

        # 关键：以“真实棋盘状态”为准，避免 turn_number 口径不一致导致视角错乱
        game.current_player = infer_current_player(board, game.board)
        game.last_move = infer_last_move(board, last_opponent_move)
        move_number = infer_move_number(board, game.board)

        pi = self.mcts.run(game, move_number)
        action = int(np.argmax(pi))
        r, c = divmod(action, self.board_size)
        return (r, c)


