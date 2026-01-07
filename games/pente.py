# pente.py
import numpy as np
from typing import List, Tuple, Optional

class Pente:
    """
    Implementação otimizada do jogo Pente para AlphaZero/MCTS.
    - Vitória: 5 em linha OU 5 pares capturados.
    - Capturas: minha - inimigo - inimigo - minha remove 2 peças e soma 1 par.
    """

    def __init__(self, size: int = 15):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = 1
        self.last_move: Optional[Tuple[int, int]] = None

        # nº de capturas por jogador
        self.captures = {1: 0, 2: 0}

        # histórico para permitir undo (necessário para MCTS sem clones)
        self.move_history: List[Tuple[int, int]] = []
        self.capture_history: List[List[Tuple[int, int]]] = []  # lista de listas

    # -------------------------------------------------------
    #  PROPRIEDADES / MAPEAMENTOS
    # -------------------------------------------------------
    @property
    def action_size(self) -> int:
        return self.size * self.size

    def action_to_move(self, action: int) -> Tuple[int, int]:
        r = action // self.size
        c = action % self.size
        return (r, c)

    def move_to_action(self, move: Tuple[int, int]) -> int:
        r, c = move
        return int(r * self.size + c)

    # -------------------------------------------------------
    #  CLONE
    # -------------------------------------------------------
    def clone(self) -> "Pente":
        new = Pente(self.size)
        new.board = self.board.copy()
        new.current_player = int(self.current_player)
        new.last_move = None if self.last_move is None else tuple(self.last_move)
        new.captures = dict(self.captures)
        new.move_history = list(self.move_history)
        new.capture_history = [list(lst) for lst in self.capture_history]
        return new

    # -------------------------------------------------------
    #  EXECUTAR JOGADA
    # -------------------------------------------------------
    def do_move(self, move: Tuple[int, int]) -> bool:
        r, c = move
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        if self.board[r, c] != 0:
            return False

        player = self.current_player
        opponent = 3 - player

        self.board[r, c] = player
        self.last_move = (r, c)

        # guardar histórico para undo
        self.move_history.append((r, c))

        # detectar capturas
        captured = self._handle_captures(r, c, player)
        self.capture_history.append(captured)

        # troca turno
        self.current_player = opponent
        return True

    # -------------------------------------------------------
    #  DESFAZER JOGADA (essencial para MCTS sem custo elevado)
    # -------------------------------------------------------
    def undo_move(self) -> None:
        if not self.move_history:
            return

        # reverte troca de jogador
        self.current_player = 3 - self.current_player

        # recuperar info do histórico
        move = self.move_history.pop()
        captured = self.capture_history.pop()

        # desfazer a jogada
        r, c = move
        self.board[r, c] = 0

        # desfazer capturas
        if captured:
            # cada captura contém 2 coordenadas de peças capturadas
            for rr, cc in captured:
                self.board[rr, cc] = self.current_player  # quem capturou agora é self.current_player

            # subtrair número de pares capturados
            self.captures[self.current_player] -= len(captured) // 2

        # atualizar last_move
        self.last_move = self.move_history[-1] if self.move_history else None

    # -------------------------------------------------------
    #  CAPTURAS
    # -------------------------------------------------------
    def _handle_captures(self, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        """
        Detecta e aplica capturas.
        Retorna lista [(r1,c1), (r2,c2), ...] das peças capturadas (para undo).
        """
        opponent = 3 - player
        size = self.size

        captured_positions = []

        directions = [
            (1, 0), (-1, 0),
            (0, 1), (0, -1),
            (1, 1), (-1, -1),
            (1, -1), (-1, 1),
        ]

        for dr, dc in directions:
            r1, c1 = r + dr, c + dc
            r2, c2 = r + 2*dr, c + 2*dc
            r3, c3 = r + 3*dr, c + 3*dc

            if (
                0 <= r1 < size and 0 <= c1 < size and
                0 <= r2 < size and 0 <= c2 < size and
                0 <= r3 < size and 0 <= c3 < size
            ):
                if (
                    self.board[r1, c1] == opponent and
                    self.board[r2, c2] == opponent and
                    self.board[r3, c3] == player
                ):
                    # captura válida
                    self.board[r1, c1] = 0
                    self.board[r2, c2] = 0
                    self.captures[player] += 1
                    captured_positions.extend([(r1, c1), (r2, c2)])

        return captured_positions

    # -------------------------------------------------------
    #  MOVIMENTOS
    # -------------------------------------------------------
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        empties = np.where(self.board == 0)
        return list(zip(empties[0].tolist(), empties[1].tolist()))

    def has_legal_moves(self) -> bool:
        return np.any(self.board == 0)

    def get_valid_moves(self) -> np.ndarray:
        """
        Retorna vetor 225-dim (ou size*size) com 1.0 onde é legal jogar.
        """
        valid = np.zeros(self.action_size, dtype=np.float32)
        empties = np.where(self.board == 0)
        for r, c in zip(empties[0], empties[1]):
            valid[self.move_to_action((int(r), int(c)))] = 1.0
        return valid

    # -------------------------------------------------------
    #  ESTADO E ENCODING
    # -------------------------------------------------------
    def get_state(self) -> np.ndarray:
        return self.board.copy()

    def get_encoded_state(self) -> np.ndarray:
        """
        Codificação estilo AlphaZero: [3, size, size]
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

    # -------------------------------------------------------
    #  VITÓRIA
    # -------------------------------------------------------
    def check_winner(self) -> int:
        last = self.last_move
        if last is None:
            return 0

        r, c = last
        player = int(self.board[r, c])
        size = self.size

        # vitória por capturas
        if self.captures[player] >= 5:
            return player

        # vitória por 5 em linha
        directions = [(1,0), (0,1), (1,1), (1,-1)]

        for dr, dc in directions:
            count = 1

            nr, nc = r + dr, c + dc
            while 0 <= nr < size and 0 <= nc < size and self.board[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc

            nr, nc = r - dr, c - dc
            while 0 <= nr < size and 0 <= nc < size and self.board[nr, nc] == player:
                count += 1
                nr -= dr
                nc -= dc

            if count >= 5:
                return player

        return 0

    def is_game_over(self) -> bool:
        return self.check_winner() != 0 or not self.has_legal_moves()

    def get_winner(self) -> int:
        return self.check_winner()

    # -------------------------------------------------------
    #  DISPLAY
    # -------------------------------------------------------
    def display(self) -> None:
        RED = "\033[31m"
        BLUE = "\033[34m"
        RESET = "\033[0m"

        print()
        col_numbers = "    " + " ".join(f"{i+1:2}" for i in range(self.size))
        print(col_numbers)

        for r in range(self.size):
            row = f"{r+1:2}  "
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
        print(f"Jogador atual: {RED}●{RESET}" if self.current_player == 1 else f"Jogador atual: {BLUE}●{RESET}")
        print(f"Capturas — {RED}●{RESET}: {self.captures[1]}   {BLUE}●{RESET}: {self.captures[2]}")
