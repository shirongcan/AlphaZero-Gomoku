import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from mcts.mcts_pure import MCTSGomoku, MCTSPente
from games.pente import Pente
from games.gomoku import Gomoku
from players.utils import infer_current_player, infer_last_move

class Player:

    def __init__(self, rules="gomoku", board_size=15, n_playout=25, c_puct=1.4):
        
        self.rules = rules.lower()
        self.board_size = board_size
        self.n_playout = n_playout

        # escolher o MCTS / heurísticas mais adequadas conforme o tipo de jogo que for
        if (self.rules == "gomoku"):
            self.mcts = MCTSGomoku(n_playout=n_playout, c_puct=c_puct)
        else:
            self.mcts = MCTSPente(n_playout=n_playout, c_puct=c_puct)

    def play(self, board, turn_number, last_opponent_move):

        # importa o jogo correto
        if self.rules == "pente":
            game = Pente(size=self.board_size)
        else:
            game = Gomoku(size=self.board_size)

        # copia o estado atual do tabuleiro
        if isinstance(board, list):
            game.board = np.array(board, dtype=int)
        else:
            game.board = np.copy(board.board)

        # 关键：以“真实棋盘状态”为准，避免 turn_number 口径不一致导致视角错乱
        game.current_player = infer_current_player(board, game.board)
        game.last_move = infer_last_move(board, last_opponent_move)

        # procura a melhor jogada via MCTS
        move = self.mcts.get_move(game)
        return move