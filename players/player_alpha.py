import numpy as np
import torch
from mcts.new_mcts_alpha import MCTS
from network import PyTorchModel
from games.pente import Pente
from games.gomoku import Gomoku
from players.utils import infer_current_player, infer_last_move, infer_move_number

class Player:
    def __init__(self,
                 rules="gomoku",
                 board_size=15,
                 n_simulations=3000,
                 c_puct=1.0,
                 model_path="models/snapshot_iter16_20260107_223122.pt",
                 nn_model=PyTorchModel):
        
        self.rules = rules.lower()
        self.board_size = board_size
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.model_path = model_path

        # 1) Criar o modelo neural (usa PyTorchModel)
        self.net = nn_model(board_size=self.board_size)

        if model_path is not None:
            print(f"[PlayerAlpha] Carregando o modelo: {model_path}")
            self.net.load(model_path)  # Usar o método load
        else:
            print("[PlayerAlpha] AVISO: Nenhum modelo fornecido. Usando pesos aleatórios!")

        # 2) Colocar o modelo em modo de avaliação
        self.net.net.eval()  # Usando o eval() no AlphaZeroNet dentro do PyTorchModel

        # 3) Definir o jogo (Gomoku ou Pente)
        if self.rules == "pente":
            self.game_class = Pente
        else:
            self.game_class = Gomoku

        # 4) Criar o MCTS
        self.mcts = MCTS(
            game_class=self.game_class,
            n_simulations=self.n_simulations,
            nn_model=self.net,  # Passa o PyTorchModel para o MCTS
            cpuct=self.c_puct,
            add_dirichlet_noise=False
        )

    # ------------------------------------------------------------
    #   JOGAR
    # ------------------------------------------------------------
    def play(self, board, turn_number, last_opponent_move):
        """
        Realiza uma jogada com base no estado atual do jogo, utilizando o MCTS e a rede neural para calcular a jogada.
        """
        # Cria o jogo
        game = self.game_class(size=self.board_size)

        # Copia o tabuleiro para o jogo
        if isinstance(board, list):
            game.board = np.array(board, dtype=int)
        else:
            game.board = np.copy(board.board)

        # 关键：以“真实棋盘状态”为准，避免 turn_number 口径不一致导致视角错乱
        game.current_player = infer_current_player(board, game.board)
        game.last_move = infer_last_move(board, last_opponent_move)
        move_number = infer_move_number(board, game.board)

        # Executa o MCTS para obter a política (probabilidades das jogadas)
        pi = self.mcts.run(game, move_number)

        # Escolhe a melhor ação com base na política
        action = np.argmax(pi)
        r, c = divmod(action, self.board_size)

        return (r, c)
