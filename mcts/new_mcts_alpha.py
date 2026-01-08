import numpy as np
import math

class MCTS:
    """
    Monte Carlo Tree Search para AlphaZero com batch prediction.
    - Adaptado aos jogos Gomoku e Pente.
    - Usa política + valor da rede neural.
    - Suporta simetrias (rotação + flip horizontal).
    """

    def __init__(self, game_class, n_simulations, nn_model, cpuct=1.0, batch_size=32, dirichlet_alpha=0.03, epsilon=0.03, apply_dirichlet_n_first_moves=10, add_dirichlet_noise=True):
        self.game_class = game_class
        self.n_simulations = n_simulations
        self.nn_model = nn_model
        self.cpuct = cpuct
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.epsilon = epsilon
        self.apply_dirichlet_n_first_moves = apply_dirichlet_n_first_moves
        self.add_dirichlet_noise = add_dirichlet_noise

        # Tree dictionaries
        self.P = {}
        self.V = {}
        self.N = {}
        self.W = {}
        self.children = {}
        self.action_size = game_class().size ** 2

        # Batch prediction queue
        self.pending_states = []
        self.pending_keys = []
        self.pending_game_states = []

        # Root node key (for Dirichlet noise)
        self.root_key = None

    # -------------------------
    #  SIMETRIAS
    # -------------------------
    def symmetries(self, state, pi):
        size = state.shape[1]
        pi = pi.reshape(size, size)

        out = []
        for k in range(4):
            rotated_s = np.rot90(state, k, axes=(1, 2))
            rotated_pi = np.rot90(pi, k)
            out.append((rotated_s, rotated_pi.flatten()))

            flipped_s = np.flip(rotated_s, axis=2)
            flipped_pi = np.flip(rotated_pi, axis=1)
            out.append((flipped_s, flipped_pi.flatten()))

        return out

    def clear_tree(self):
        # Tree dictionaries
        self.P = {}
        self.V = {}
        self.N = {}
        self.W = {}
        self.children = {}

        # Batch prediction queue
        self.pending_states = []
        self.pending_keys = []
        self.pending_game_states = []

        # Root node key (for Dirichlet noise)
        self.root_key = None

    # -------------------------
    #  MCTS RUN
    # -------------------------
    def run(self, game_state, move_number):
        self.root_key = self._state_key(game_state)

        for _ in range(self.n_simulations):
            game_copy = game_state.clone()
            self.search(game_copy, move_number)

        # processar quaisquer estados remanescentes
        self._predict_batch(move_number)

        s_key = self._state_key(game_state)
        counts = self.N[s_key]
        valid = self.children[s_key]
        
        # 确保只考虑合法动作
        counts = counts * valid
        total = np.sum(counts)
        
        if total > 0:
            pi = counts / total
        else:
            # fallback seguro: uniformemente entre ações válidas
            pi = valid / (np.sum(valid) + 1e-8)
        
        # 再次确保只包含合法动作并归一化（安全措施）
        pi = pi * valid
        pi = pi / (np.sum(pi) + 1e-8)
        
        return pi

    # -------------------------
    #  SEARCH
    # -------------------------
    def search(self, game_state, move_number):
        s_key = self._state_key(game_state)

        # Verificar se o jogo terminou (ganhador ou sem movimentos válidos)
        if game_state.is_game_over():
            winner = game_state.get_winner()
            if winner == 0:
                return 0
            return 1 if winner == game_state.current_player else -1

        if s_key not in self.P:
            # adiciona à fila de batch
            self.pending_states.append(game_state.get_encoded_state())
            self.pending_keys.append(s_key)
            self.pending_game_states.append(game_state.clone())  # <--- armazenar o estado completo

            # se a fila encheu, processa em batch
            if len(self.pending_states) >= self.batch_size:
                self._predict_batch(move_number)

            # fallback: se ainda não foi processado, inicializa uniformemente
            if s_key not in self.P:
                valid = game_state.get_valid_moves()
                self.P[s_key] = valid / np.sum(valid)
                self.V[s_key] = 0
                self.N[s_key] = np.zeros_like(valid, dtype=np.float32)
                self.W[s_key] = np.zeros_like(valid, dtype=np.float32)
                self.children[s_key] = valid
                return self.V[s_key]

        # já visitado → UCB
        valid = self.children[s_key]
        sqrt_sum = math.sqrt(np.sum(self.N[s_key]))
        ucb = self.W[s_key] / (1 + self.N[s_key]) + self.cpuct * self.P[s_key] * sqrt_sum / (1 + self.N[s_key])
        ucb = np.where(valid == 1, ucb, -1e9)

        action = np.argmax(ucb)
        r, c = divmod(action, game_state.size)
        
        # Realizar a jogada, tratando as capturas no caso do Pente
        game_state.do_move((r, c))  # Aqui é onde a captura de peças será tratada

        v = -self.search(game_state, move_number)

        self.W[s_key][action] += v
        self.N[s_key][action] += 1

        return v

    # -------------------------
    #  BATCH PREDICTION
    # -------------------------
    def _predict_batch(self, move_number):
        if not self.pending_states:
            return

        X = np.stack(self.pending_states, axis=0).astype(np.float32)
        policies, values = self.nn_model.predict(X)

        for k, p, v, gs in zip(self.pending_keys, policies, values, self.pending_game_states):
            p = p.flatten()
            valid = gs.get_valid_moves()  # <-- usa o estado real armazenado na fila
            p = p * valid
            if np.sum(p) < 1e-8:
                p = valid / np.sum(valid)

            # Add Dirichlet noise ONLY at the root node
            if self.add_dirichlet_noise and k == self.root_key and move_number < self.apply_dirichlet_n_first_moves:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(p))
                p = (1 - self.epsilon) * p + self.epsilon * noise
                p /= np.sum(p)

            self.P[k] = p
            self.V[k] = v[0] if hasattr(v, "__len__") else v
            self.N[k] = np.zeros_like(p, dtype=np.float32)
            self.W[k] = np.zeros_like(p, dtype=np.float32)
            self.children[k] = valid

        # limpar fila
        self.pending_states = []
        self.pending_keys = []
        self.pending_game_states = []

    # -------------------------
    #  STATE KEY
    # -------------------------
    def _state_key(self, game_state):
        board = game_state.board
        player = game_state.current_player
        return board.tobytes() + bytes([player])