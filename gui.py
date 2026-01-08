# play.py
import sys
import importlib
import time
import os
import subprocess

from games.gomoku import Gomoku

# =========================================================================== #
#               Para jogar usando o play.py digite no terminal:               #
#                  python play.py player_human player_human                   #
# =========================================================================== #

RED = "\033[31m"
BLUE = "\033[34m"
RESET = "\033[0m"

# ====== DYNAMIC PLAYER LOADING ======
def load_player(module_name, rules, size):
    # aceita 'player_mcts' ou 'player_mcts.py'
    module_name = module_name.replace(".py", "").strip()
    if not module_name.startswith("players."):
        module_name = f"players.{module_name}"

    module = importlib.import_module(module_name)

    if hasattr(module, "Player"):
        return module.Player(rules, size)

    raise ValueError(f"Nenhuma classe Player encontrada em {module_name}")


# ====== GAME MENU ======
def choose_game():
    # Projeto agora suporta apenas Gomoku
    return "gomoku"

# ====== MAIN ======
def main():
    if len(sys.argv) != 3:
        print("Uso: python gui.py <player1> <player2>")
        print("Exemplo: python gui.py player_human player_mcts")
        sys.exit(1)

    player1_name, player2_name = sys.argv[1:3]

    # Menu interativo
    game_name = choose_game()
    size = 15

    # Inicializa o jogo
    game = Gomoku(size)

    # Carrega os jogadores
    player1 = load_player(player1_name, game_name, size)
    player2 = load_player(player2_name, game_name, size)
    players = {1: player1, 2: player2}

    # Detectar se algum dos players é humano (pelo nome do módulo)
    def is_human_player(name: str) -> bool:
        base = name.replace(".py", "").strip().lower()
        return "human" in base  # "player_human"

    p1_is_human = is_human_player(player1_name)
    p2_is_human = is_human_player(player2_name)

    # === LOG DE JOGADAS PARA A INTERFACE-ESPELHO ===
    log_path = "mirror_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"GAME {game_name}\n")
    log_file.flush()

    # === LOG DE INPUT PARA MOVIMENTOS HUMANOS PELA INTERFACE ===
    input_path = "input_log.txt"
    # limpa o ficheiro no início
    with open(input_path, "w", encoding="utf-8"):
        pass
    last_input_lines = 0

    print(f"\n🎮 Iniciando {game_name.capitalize()}")
    print(f"{RED}●{RESET} Jogador 1: {player1_name} ({'HUMANO' if p1_is_human else 'BOT'})")
    print(f"{BLUE}●{RESET} Jogador 2: {player2_name} ({'HUMANO' if p2_is_human else 'BOT'})\n")

    game.display()

    # === LANÇAR AUTOMATICAMENTE A INTERFACE PYGAME EM MODO ESPELHO ===
    try:
        script_path = os.path.join(os.path.dirname(__file__), "interface.py")
        subprocess.Popen([
            sys.executable,
            script_path,
            "mirror",
            game_name,
            "human" if p1_is_human else "bot",
            "human" if p2_is_human else "bot",
        ])
    except Exception as e:
        print(f"(Aviso) Não foi possível iniciar a interface gráfica: {e}")

    turn_number = 0

    # Loop principal
    while not game.is_game_over():
        current_player = game.current_player
        is_human_turn = (current_player == 1 and p1_is_human) or (current_player == 2 and p2_is_human)

        valid_move = False
        while not valid_move:
            if is_human_turn:
                # === LER JOGADA DA INTERFACE (input_log.txt) ===
                print(f"Aguardando jogada do interface para o Jogador {current_player}...")
                move = None
                while move is None:
                    try:
                        with open(input_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                    except FileNotFoundError:
                        lines = []

                    if len(lines) > last_input_lines:
                        line = lines[last_input_lines].strip()
                        last_input_lines += 1
                        parts = line.split()
                        if len(parts) == 2:
                            try:
                                r = int(parts[0])
                                c = int(parts[1])
                                move = (r, c)
                            except ValueError:
                                move = None

                    if move is None:
                        time.sleep(0.05)  # espera um bocadinho e volta a ler

            else:
                # === JOGADA DO BOT (ou outro player não-humano) ===
                player = players[current_player]
                start_time = time.time()
                try:
                    move = player.play(game.clone(), turn_number, game.last_move)
                except Exception as e:
                    print(f"Erro no jogador {current_player}: {e}")
                    continue

                think_time = time.time() - start_time
                print(f"⏱️  Tempo de decisão: {think_time:.2f}s")

                if move is None:
                    print("Jogador não devolveu jogada. Tente novamente.")
                    continue

            # TENTAR APLICAR A JOGADA (tanto de humano como de bot)
            try:
                game.do_move(move)

                # registar jogada no log de espelho
                r, c = move
                log_file.write(f"{r} {c}\n")
                log_file.flush()

                valid_move = True
                turn_number += 1
            except ValueError as e:
                if is_human_turn:
                    print(f"Jogada inválida recebida da interface: {e}")
                else:
                    print(f"Jogada inválida: {e}")
                # volta ao while not valid_move

        game.display()  # mostra o tabuleiro após cada jogada válida

    # Certifica que o último estado do tabuleiro é mostrado
    print("\nEstado final do tabuleiro:")
    game.display()

    # Mensagem de vitória
    winner = game.get_winner()
    if winner == 0:
        print("\nEmpate! Nenhum vencedor.")
    else:
        if winner == 1:
            piece = f"{RED}●{RESET}"
        else:
            piece = f"{BLUE}●{RESET}"

        print(f"\n🏆 Jogador {winner} ({piece}) venceu!")

    # Fecha o ficheiro de log
    log_file.close()

if __name__ == "__main__":
    main()