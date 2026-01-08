import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from mcts.mcts_pure import MCTSGomoku
from games.gomoku import Gomoku

class Player:

    def __init__(self, rules="gomoku", board_size=15, n_playout=25, c_puct=1.4):
        
        self.rules = rules.lower()
        self.board_size = board_size
        self.n_playout = n_playout

        # Projeto agora suporta apenas Gomoku
        if self.rules != "gomoku":
            raise ValueError(f"Unsupported rules: {self.rules}. Only 'gomoku' is supported.")
        self.mcts = MCTSGomoku(n_playout=n_playout, c_puct=c_puct)

    def play(self, board, turn_number, last_opponent_move):

        # importa o jogo correto (apenas Gomoku)
        game = Gomoku(size=self.board_size)

        # copia o estado atual do tabuleiro
        if isinstance(board, list):
            game.board = np.array(board, dtype=int)
        else:
            game.board = np.copy(board.board)

        # define jogador atual
        game.current_player = 1 if turn_number % 2 == 0 else 2
        game.last_move = last_opponent_move

        # procura a melhor jogada via MCTS
        move = self.mcts.get_move(game)
        return move