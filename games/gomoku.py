# gomoku.py
import numpy as np
from typing import List, Tuple, Optional

class Gomoku:
    """
    Implementação do jogo Gomoku (15x15 por defeito).
    - Vitória: 5 em linha (horizontal, vertical, diagonal)
    - Estado interno:
        board : np.array (size x size) com {0: vazio, 1: jogador1, 2: jogador2}
        current_player : 1 ou 2
        move_history : lista de (r,c) jogadas aplicadas na ordem
        last_move : última jogada (r,c) ou None
    Notas para AlphaZero/MCTS:
    - action_size = size * size
    - get_valid_moves() retorna vetor binário (float32) de dimensão action_size
    - action index = r * size + c
    """

    def __init__(self, size: int = 15):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = 1
        self.move_history: List[Tuple[int, int]] = []
        self.last_move: Optional[Tuple[int, int]] = None

    # ----------------------
    # clonagem / cópia
    # ----------------------
    def clone(self) -> "Gomoku":
        """Retorna uma cópia profunda do jogo (útil para debugging / simulações isoladas)."""
        new_game = Gomoku(self.size)
        new_game.board = self.board.copy()
        new_game.current_player = int(self.current_player)
        new_game.move_history = list(self.move_history)
        new_game.last_move = None if self.last_move is None else tuple(self.last_move)
        return new_game

    # ----------------------
    # utilitários ação <-> movimento
    # ----------------------
    @property
    def action_size(self) -> int:
        return self.size * self.size

    def action_to_move(self, action: int) -> Tuple[int, int]:
        """Converte action index (0..size*size-1) para (r,c)."""
        r = action // self.size
        c = action % self.size
        return (r, c)

    def move_to_action(self, move: Tuple[int, int]) -> int:
        """Converte (r,c) para action index."""
        r, c = move
        return int(r * self.size + c)

    # ----------------------
    # jogadas / undo
    # ----------------------
    def do_move(self, move: Tuple[int, int]) -> bool:
        """
        Executa uma jogada.
        - move: (r,c)
        - retorna True se jogada for válida e aplicada, False caso contrário.
        """
        r, c = move
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        if self.board[r, c] != 0:
            return False

        # aplica jogada
        self.board[r, c] = self.current_player
        self.move_history.append((r, c))
        self.last_move = (r, c)
        # troca jogador
        self.current_player = 3 - self.current_player
        return True

    def undo_move(self) -> None:
        """
        Desfaz a última jogada.
        - Reverte board, current_player e last_move.
        - Se não houver jogadas, não faz nada.
        """
        if not self.move_history:
            return

        last = self.move_history.pop()  # (r, c)
        r, c = last
        # limpa a casa
        self.board[r, c] = 0
        # reverte jogador (já tinha sido trocado em do_move)
        self.current_player = 3 - self.current_player
        # actualiza last_move para a nova última jogada (ou None)
        self.last_move = self.move_history[-1] if self.move_history else None

    # ----------------------
    # movimentos legais / válidos
    # ----------------------
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Lista de coordenadas vazias (útil para interface/humano)."""
        empties = np.where(self.board == 0)
        return list(zip(empties[0].tolist(), empties[1].tolist()))

    def has_legal_moves(self) -> bool:
        return np.any(self.board == 0)

    def get_valid_moves(self) -> np.ndarray:
        """
        Retorna um vetor binário (float32) de dimensão action_size.
        - 1.0 onde a ação é legal, 0.0 onde é ilegal.
        Este formato é o que a rede + MCTS esperam (policy mask).
        """
        valid = np.zeros(self.action_size, dtype=np.float32)
        empties = np.where(self.board == 0)
        # zip devolve pares (r,c)
        for r, c in zip(empties[0], empties[1]):
            idx = self.move_to_action((int(r), int(c)))
            valid[idx] = 1.0
        return valid

    # ----------------------
    # estado / codificação para rede
    # ----------------------
    def get_state(self) -> np.ndarray:
        """Retorna uma cópia simples do tabuleiro (size x size)."""
        return self.board.copy()

    def get_encoded_state(self) -> np.ndarray:
        """
        Retorna o estado codificado para a rede AlphaZero: np.array [C, H, W]
        C = 3:
          - canal 0: peças do jogador atual (1.0/0.0)
          - canal 1: peças do jogador oponente (1.0/0.0)
          - canal 2: turno (1.0 se current_player == 1, 0.0 se current_player == 2)
        Esta codificação é perspectiva-invariant: sempre do ponto de vista do jogador atual.
        """
        board = self.board
        size = self.size

        player = self.current_player
        opponent = 3 - player

        p_cur = (board == player).astype(np.float32)
        p_opp = (board == opponent).astype(np.float32)
        # turn通道：1.0表示玩家1，0.0表示玩家2
        turn_value = 1.0 if player == 1 else 0.0
        turn = np.full((size, size), turn_value, dtype=np.float32)

        return np.stack([p_cur, p_opp, turn], axis=0)

    # ----------------------
    # verificação de vitória (usa last_move para eficiência)
    # ----------------------
    def check_winner(self) -> int:
        """
        Verifica vencedor considerando apenas a última jogada (muito mais rápido).
        Retorna:
          0 -> sem vencedor
          1 -> jogador 1 venceu
          2 -> jogador 2 venceu
        """
        if self.last_move is None:
            return 0

        r, c = self.last_move
        player = int(self.board[r, c])
        if player == 0:
            return 0

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            # direção positiva
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc

            # direção negativa
            nr, nc = r - dr, c - dc
            while 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == player:
                count += 1
                nr -= dr
                nc -= dc

            if count >= 5:
                return player

        return 0

    def is_game_over(self) -> bool:
        """True se houver vencedor ou não houver jogadas legais."""
        return self.check_winner() != 0 or not self.has_legal_moves()

    def get_winner(self) -> int:
        """Retorna o vencedor (0 se não houver)."""
        return self.check_winner()

    # ----------------------
    # visualização (mantive a tua função)
    # ----------------------
    def display(self) -> None:
        """Mostra o tabuleiro no terminal, com cores e alinhamento."""
        RED = "\033[31m"   # vermelho
        BLUE = "\033[34m"  # azul
        RESET = "\033[0m"  # reset de cor

        print()

        # Cabeçalho de colunas (alinhado)
        col_numbers = "    " + " ".join(f"{i+1:2}" for i in range(self.size))
        print(col_numbers)

        # Corpo do tabuleiro
        for r in range(self.size):
            row = f"{r+1:2}  "  # número da linha + espaçamento
            for c in range(self.size):
                val = int(self.board[r, c])
                if val == 1:
                    row += f" {RED}●{RESET} "
                elif val == 2:
                    row += f" {BLUE}●{RESET} "
                else:
                    row += " - "
            print(row)

        print()
        current = "1" if self.current_player == 1 else "2"
        color_dot = f"{RED}●{RESET}" if self.current_player == 1 else f"{BLUE}●{RESET}"
        print(f"Jogador atual: {color_dot} (player {current})")
