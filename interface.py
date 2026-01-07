import pygame
import sys
import importlib
import math

from games.gomoku import Gomoku
from games.pente import Pente

WIDTH, HEIGHT = 900, 720
FPS = 60

BG_COLOR = (18, 12, 24)
PANEL_COLOR = (45, 35, 70)
BOARD_BG_COLOR = (248, 235, 205)
GRID_COLOR = (70, 50, 30)
HIGHLIGHT_COLOR = (255, 230, 120)

BLACK_STONE = (25, 25, 30)
WHITE_STONE = (245, 245, 245)

TEXT_COLOR = (235, 235, 235)
ACCENT_COLOR = (140, 110, 190)
BUTTON_HOVER = (75, 65, 110)
BUTTON_SELECTED = (140, 110, 190)

BOARD_SIZE = 15
BOARD_PIXEL_SIZE = 600
CELL_SIZE = BOARD_PIXEL_SIZE // BOARD_SIZE
BOARD_ORIGIN_X = (WIDTH - BOARD_PIXEL_SIZE) // 2
BOARD_ORIGIN_Y = 100

STATE_GAME_SELECT = "game_select"
STATE_PLAYER_SELECT = "player_select"
STATE_PLAYING = "playing"

MENU_BG_IMAGE_PATH = "interface_menus/menu3.webp"


def get_chinese_font(size, bold=False):
    """
    获取支持中文的字体。尝试多个中文字体，如果都失败则使用系统默认字体。
    """
    # Windows系统常见的中文字体列表（按优先级排序）
    chinese_fonts = [
        "Microsoft YaHei",  # 微软雅黑
        "SimHei",           # 黑体
        "SimSun",           # 宋体
        "KaiTi",            # 楷体
        "FangSong",         # 仿宋
    ]
    
    # 尝试加载中文字体
    for font_name in chinese_fonts:
        try:
            font = pygame.font.SysFont(font_name, size, bold=bold)
            # 测试字体是否能渲染中文
            test_surf = font.render("测试", True, (255, 255, 255))
            if test_surf.get_width() > 0:  # 如果能渲染，说明字体支持中文
                return font
        except Exception:
            continue
    
    # 如果所有中文字体都失败，尝试使用系统默认字体
    try:
        return pygame.font.SysFont(None, size, bold=bold)
    except Exception:
        # 最后的回退：使用pygame默认字体
        return pygame.font.Font(None, size)


class HumanGUIPlayer:
    def __init__(self, name="Human"):
        self.name = name
        self.pending_move = None

    def set_click(self, move):
        self.pending_move = move

    def play(self, board, turn_number, last_move):
        if self.pending_move is None:
            return None
        move = self.pending_move
        self.pending_move = None
        return move


class Button:
    def __init__(self, rect, text, font, callback, data=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.callback = callback
        self.data = data
        self.selected = False

    def draw(self, surface, mouse_pos):
        is_hover = self.rect.collidepoint(mouse_pos)

        # rect para desenho (pode inflar para animação)
        draw_rect = self.rect.copy()
        if is_hover:
            draw_rect.inflate_ip(6, 4)

        # sombra
        shadow_rect = draw_rect.copy()
        shadow_rect.x += 3
        shadow_rect.y += 4
        pygame.draw.rect(surface, (0, 0, 0, 80), shadow_rect, border_radius=12)

        # cor principal
        if self.selected:
            color = BUTTON_SELECTED
        elif is_hover:
            color = BUTTON_HOVER
        else:
            color = PANEL_COLOR

        pygame.draw.rect(surface, color, draw_rect, border_radius=12)
        pygame.draw.rect(surface, GRID_COLOR, draw_rect, 2, border_radius=12)

        text_surf = self.font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=draw_rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback(self.data)


def load_bot_player(module_name, rules, size):
    module_name = module_name.replace(".py", "").strip()
    if not module_name.startswith("players."):
        module_name = f"players.{module_name}"

    module = importlib.import_module(module_name)
    if hasattr(module, "Player"):
        return module.Player(rules, size)
    raise ValueError(f"No class Player found in {module_name}")


def draw_centered_text(surface, text, font, y, color=TEXT_COLOR, pulse=0.0):
    """pulse ∈ [0,1] aumenta ligeiramente o tamanho do texto"""
    text_surf = font.render(text, True, color)
    if pulse > 0:
        scale = 1.0 + 0.06 * pulse
        w, h = text_surf.get_size()
        new_size = (int(w * scale), int(h * scale))
        text_surf = pygame.transform.smoothscale(text_surf, new_size)
    text_rect = text_surf.get_rect(center=(WIDTH // 2, y))
    surface.blit(text_surf, text_rect)


def cell_center(row, col):
    cx = BOARD_ORIGIN_X + CELL_SIZE // 2 + col * CELL_SIZE
    cy = BOARD_ORIGIN_Y + CELL_SIZE // 2 + row * CELL_SIZE
    return cx, cy


def draw_board(surface, game):
    board_rect = pygame.Rect(
        BOARD_ORIGIN_X, BOARD_ORIGIN_Y, BOARD_PIXEL_SIZE, BOARD_PIXEL_SIZE
    )

    shadow_rect = board_rect.copy()
    shadow_rect.topleft = (shadow_rect.left + 4, shadow_rect.top + 6)
    pygame.draw.rect(surface, (10, 8, 18), shadow_rect, border_radius=16)

    pygame.draw.rect(surface, BOARD_BG_COLOR, board_rect, border_radius=16)
    pygame.draw.rect(surface, GRID_COLOR, board_rect, 3, border_radius=16)

    for i in range(BOARD_SIZE):
        start = (BOARD_ORIGIN_X + CELL_SIZE // 2,
                 BOARD_ORIGIN_Y + CELL_SIZE // 2 + i * CELL_SIZE)
        end = (BOARD_ORIGIN_X + BOARD_PIXEL_SIZE - CELL_SIZE // 2,
               BOARD_ORIGIN_Y + CELL_SIZE // 2 + i * CELL_SIZE)
        pygame.draw.line(surface, GRID_COLOR, start, end, 1)

        start = (BOARD_ORIGIN_X + CELL_SIZE // 2 + i * CELL_SIZE,
                 BOARD_ORIGIN_Y + CELL_SIZE // 2)
        end = (BOARD_ORIGIN_X + CELL_SIZE // 2 + i * CELL_SIZE,
               BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE - CELL_SIZE // 2)
        pygame.draw.line(surface, GRID_COLOR, start, end, 1)

    board = game.board
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = int(board[r, c])
            if v == 0:
                continue
            cx, cy = cell_center(r, c)
            color = BLACK_STONE if v == 1 else WHITE_STONE
            pygame.draw.circle(surface, color, (cx, cy), CELL_SIZE // 2 - 3)


def draw_last_move_ring(surface, game):
    last_move = getattr(game, "last_move", None)
    if last_move is None:
        return
    r, c = last_move
    cx, cy = cell_center(r, c)
    radius = CELL_SIZE // 2 - 2
    pygame.draw.circle(surface, HIGHLIGHT_COLOR, (cx, cy), radius, 3)


def draw_ghost_stone(surface, game, current_player, mouse_pos):
    pos = screen_to_board(mouse_pos)
    if pos is None:
        return
    r, c = pos
    if game.board[r, c] != 0:
        return

    cx, cy = cell_center(r, c)
    ghost_color = BLACK_STONE if current_player == 1 else WHITE_STONE

    ghost_surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(
        ghost_surf,
        ghost_color + (120,),
        (CELL_SIZE // 2, CELL_SIZE // 2),
        CELL_SIZE // 2 - 3,
    )
    surface.blit(ghost_surf, (cx - CELL_SIZE // 2, cy - CELL_SIZE // 2))


def screen_to_board(pos):
    x, y = pos
    if not (BOARD_ORIGIN_X <= x < BOARD_ORIGIN_X + BOARD_PIXEL_SIZE and
            BOARD_ORIGIN_Y <= y < BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE):
        return None

    col = (x - BOARD_ORIGIN_X) // CELL_SIZE
    row = (y - BOARD_ORIGIN_Y) // CELL_SIZE
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return (row, col)
    return None


def build_replay_game(original_game, upto_moves):
    if isinstance(original_game, Gomoku):
        g = Gomoku(size=BOARD_SIZE)
    else:
        g = Pente(size=BOARD_SIZE)

    history = getattr(original_game, "move_history", [])
    upto_moves = min(upto_moves, len(history))
    for i in range(upto_moves):
        g.do_move(history[i])
    return g


def get_player_label(choice_key, idx):
    if choice_key == "human":
        return f"Human {idx}"
    elif choice_key == "mcts":
        return f"MCTS {idx}"
    elif choice_key == "alpha":
        return f"AlphaZero {idx}"
    else:
        return f"Player {idx}"


def main():
    pygame.init()
    pygame.display.set_caption("五子棋")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    title_font = get_chinese_font(70, bold=True)
    menu_font = get_chinese_font(28, bold=True)
    small_font = get_chinese_font(22)
    tiny_font = get_chinese_font(18)

    # tentar carregar imagem de fundo do menu (opcional)
    menu_bg = None
    try:
        img = pygame.image.load(MENU_BG_IMAGE_PATH)
        menu_bg = pygame.transform.smoothscale(img, (WIDTH, HEIGHT))
    except Exception:
        menu_bg = None

    state = STATE_GAME_SELECT

    selected_game = None
    selected_p1 = None
    selected_p2 = None

    game = None
    players = {}
    turn_number = 0

    buttons = []
    end_buttons = []
    replay_buttons = []
    play_buttons = []

    game_over_handled = False
    replay_index = None
    replay_game = None

    def reset_buttons():
        nonlocal buttons
        buttons = []

    def reset_end_buttons():
        nonlocal end_buttons, game_over_handled
        end_buttons = []
        game_over_handled = False

    def reset_replay():
        nonlocal replay_index, replay_game
        replay_index = None
        replay_game = None

    def go_to_player_select(game_name):
        nonlocal state, selected_game, selected_p1, selected_p2
        selected_game = game_name
        selected_p1 = None
        selected_p2 = None
        state = STATE_PLAYER_SELECT
        setup_player_select_buttons()

    def set_player1(choice_key):
        nonlocal selected_p1
        selected_p1 = choice_key
        for b in buttons:
            if getattr(b, "column", None) == 1:
                b.selected = (b.data == choice_key)

    def set_player2(choice_key):
        nonlocal selected_p2
        selected_p2 = choice_key
        for b in buttons:
            if getattr(b, "column", None) == 2:
                b.selected = (b.data == choice_key)

    def create_player(choice_key):
        if choice_key == "human":
            return HumanGUIPlayer()
        elif choice_key == "mcts":
            return load_bot_player("player_mcts", selected_game, BOARD_SIZE)
        elif choice_key == "alpha":
            return load_bot_player("player_alpha", selected_game, BOARD_SIZE)
        else:
            raise ValueError("Player type unknown")

    def setup_replay_buttons():
        nonlocal replay_buttons
        replay_buttons = []
        y = BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE // 2 - 20
        left_rect = (BOARD_ORIGIN_X - 70, y, 50, 40)
        right_rect = (BOARD_ORIGIN_X + BOARD_PIXEL_SIZE + 20, y, 50, 40)
        replay_buttons.append(Button(left_rect, "<", menu_font, lambda _: step_replay(-1)))
        replay_buttons.append(Button(right_rect, ">", menu_font, lambda _: step_replay(1)))

    def reset_play_buttons():
        nonlocal play_buttons
        play_buttons = []

    def setup_play_buttons():
        nonlocal play_buttons
        reset_play_buttons()
        btn_w, btn_h = 90, 32
        margin = 20
        x = margin
        y = HEIGHT - btn_h - margin

        play_buttons.append(Button(
            rect=(x, y, btn_w, btn_h),
            text="Menu",
            font=tiny_font,
            callback=back_to_menu
        ))

    def start_new_match():
        nonlocal game, players, turn_number, state
        reset_end_buttons()
        reset_replay()

        if selected_game == "gomoku":
            game_local = Gomoku(size=BOARD_SIZE)
        else:
            game_local = Pente(size=BOARD_SIZE)

        p1 = create_player(selected_p1)
        p2 = create_player(selected_p2)

        game = game_local
        players = {1: p1, 2: p2}
        turn_number = 0
        state = STATE_PLAYING
        setup_replay_buttons()
        setup_play_buttons()

    def start_game_if_ready(_):
        if selected_game is None or selected_p1 is None or selected_p2 is None:
            return
        start_new_match()
        reset_buttons()

    def back_to_menu(_=None):
        nonlocal state, selected_game, selected_p1, selected_p2, game
        state = STATE_GAME_SELECT
        selected_game = None
        selected_p1 = None
        selected_p2 = None
        game = None
        reset_buttons()
        reset_end_buttons()
        reset_replay()
        reset_play_buttons()
        setup_game_select_buttons()

    def step_replay(delta):
        nonlocal replay_index, replay_game
        if game is None:
            return
        history = getattr(game, "move_history", [])
        if not history:
            return

        if replay_index is None:
            if delta < 0:
                replay_index = len(history) - 1
            else:
                return
        else:
            replay_index += delta

        if replay_index < 0:
            replay_index = 0

        if replay_index >= len(history):
            replay_index = None
            replay_game = None
            return

        replay_game = build_replay_game(game, replay_index + 1)

    def setup_game_select_buttons():
        reset_buttons()
        btn_w, btn_h = 240, 60
        gap = 40
        total_w = btn_w * 2 + gap
        start_x = (WIDTH - total_w) // 2
        y = HEIGHT // 2 + 40

        buttons.append(Button(
            rect=(start_x, y, btn_w, btn_h),
            text="GOMOKU",
            font=menu_font,
            callback=lambda _: go_to_player_select("gomoku"),
            data=None
        ))

        buttons.append(Button(
            rect=(start_x + btn_w + gap, y, btn_w, btn_h),
            text="PENTE",
            font=menu_font,
            callback=lambda _: go_to_player_select("pente"),
            data=None
        ))

    def setup_player_select_buttons():
        reset_buttons()
        col_w = 260
        col_gap = 80
        total_w = col_w * 2 + col_gap
        base_x = (WIDTH - total_w) // 2
        col1_x = base_x
        col2_x = base_x + col_w + col_gap
        base_y = HEIGHT // 2 - 80
        btn_h = 50
        spacing = 10

        choices = [
            ("Human", "human"),
            ("MCTS", "mcts"),
            ("AlphaZero", "alpha"),
        ]

        for i, (label, key) in enumerate(choices):
            rect = (col1_x, base_y + i * (btn_h + spacing), col_w, btn_h)
            b = Button(rect, label, menu_font, set_player1, data=key)
            b.column = 1
            buttons.append(b)

        for i, (label, key) in enumerate(choices):
            rect = (col2_x, base_y + i * (btn_h + spacing), col_w, btn_h)
            b = Button(rect, label, menu_font, set_player2, data=key)
            b.column = 2
            buttons.append(b)

        start_rect = (WIDTH // 2 - 120, HEIGHT - 120, 240, 60)
        buttons.append(Button(start_rect, "Start Game", menu_font, start_game_if_ready))

    def setup_endgame_buttons():
        nonlocal end_buttons, game_over_handled
        end_buttons = []
        game_over_handled = True

        btn_w, btn_h = 220, 55
        gap = 40
        total_w = btn_w * 2 + gap
        start_x = (WIDTH - total_w) // 2
        y = HEIGHT // 2 + 60

        end_buttons.append(Button(
            rect=(start_x, y, btn_w, btn_h),
            text="Rematch!",
            font=menu_font,
            callback=lambda _: start_new_match(),
            data=None
        ))
        end_buttons.append(Button(
            rect=(start_x + btn_w + gap, y, btn_w, btn_h),
            text="Back to menu",
            font=menu_font,
            callback=back_to_menu,
            data=None
        ))

    setup_game_select_buttons()

    running = True
    while running:
        dt = clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()
        t = pygame.time.get_ticks() / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if state in (STATE_GAME_SELECT, STATE_PLAYER_SELECT):
                for b in buttons:
                    b.handle_event(event)

            elif state == STATE_PLAYING:
                for b in replay_buttons:
                    b.handle_event(event)

                for b in play_buttons:
                    b.handle_event(event)


                if game and not game.is_game_over() and replay_index is None:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        move = screen_to_board(event.pos)
                        if move is not None:
                            current_player = game.current_player
                            if isinstance(players[current_player], HumanGUIPlayer):
                                r, c = move
                                if game.board[r, c] == 0:
                                    players[current_player].set_click(move)
                else:
                    for b in end_buttons:
                        b.handle_event(event)

        # fundo base comum a todos os estados
        if menu_bg is not None:
            screen.blit(menu_bg, (0, 0))
        else:
            screen.fill(BG_COLOR)

       
        if state == STATE_GAME_SELECT:

            # background de imagem se existir
            if menu_bg is not None:
                screen.blit(menu_bg, (0, 0))

            pulse = (math.sin(t * 2.0) + 1) / 2  # 0..1
            draw_centered_text(screen, "五子棋", title_font, 120,
                               ACCENT_COLOR, pulse=pulse)
            draw_centered_text(screen, "选择你的游戏", menu_font, 180)

            for b in buttons:
                b.draw(screen, mouse_pos)

        elif state == STATE_PLAYER_SELECT:
            game_label = "GOMOKU" if selected_game == "gomoku" else "PENTE"
            draw_centered_text(screen, game_label, title_font, 90, ACCENT_COLOR)
            draw_centered_text(screen, "Select the players", menu_font, 140)

            screen.blit(
                small_font.render("Player 1 (Blacks)", True, TEXT_COLOR),
                (WIDTH // 2 - 260, 200),
            )
            screen.blit(
                small_font.render("Player 2 (Whites)", True, TEXT_COLOR),
                (WIDTH // 2 + 40, 200),
            )

            for b in buttons:
                b.draw(screen, mouse_pos)

        elif state == STATE_PLAYING and game is not None:
            active_game = replay_game if replay_index is not None else game

            game_label = "GOMOKU" if isinstance(game, Gomoku) else "PENTE"
            shadow_surf = title_font.render(game_label, True, (0, 0, 0))
            shadow_rect = shadow_surf.get_rect(center=(WIDTH // 2 + 2, 55 + 2))
            screen.blit(shadow_surf, shadow_rect)
            draw_centered_text(screen, game_label, title_font, 55, ACCENT_COLOR)

            if replay_index is None:
                if not game.is_game_over():
                    turn_text = f"Player {game.current_player}'s turn"
                else:
                    winner = game.get_winner()
                    if winner == 0:
                        turn_text = "Draw!"
                    else:
                        turn_text = f"Player {winner} Won!"
            else:
                history = getattr(game, "move_history", [])
                turn_text = f"Review Moves ({replay_index + 1}/{len(history)})"

            screen.blit(small_font.render(turn_text, True, TEXT_COLOR), (20, 20))

            p1_label = get_player_label(selected_p1, 1)
            p2_label = get_player_label(selected_p2, 2)
            screen.blit(tiny_font.render(f"P1: {p1_label}", True, TEXT_COLOR), (20, 50))
            screen.blit(tiny_font.render(f"P2: {p2_label}", True, TEXT_COLOR), (20, 70))

            from games.pente import Pente as PenteClass
            if isinstance(game, PenteClass):
                caps1 = game.captures.get(1, 0)
                caps2 = game.captures.get(2, 0)
                screen.blit(
                    tiny_font.render(f"P1's Captures: {caps1}", True, TEXT_COLOR),
                    (WIDTH - 200, 50),
                )
                screen.blit(
                    tiny_font.render(f"P2's Captures: {caps2}", True, TEXT_COLOR),
                    (WIDTH - 200, 70),
                )

            draw_board(screen, active_game)
            draw_last_move_ring(screen, active_game)

            for b in replay_buttons:
                b.draw(screen, mouse_pos)
            for b in play_buttons:
                b.draw(screen, mouse_pos)

            if replay_index is None and not game.is_game_over():
                current_player = game.current_player
                player_obj = players[current_player]

                if isinstance(player_obj, HumanGUIPlayer):
                    draw_ghost_stone(screen, game, current_player, mouse_pos)
                    move = player_obj.play(game, turn_number, game.last_move)
                    if move is not None and game.do_move(move):
                        turn_number += 1
                else:
                    move = player_obj.play(game, turn_number, game.last_move)
                    if move is not None and game.do_move(move):
                        turn_number += 1
            else:
                if game.is_game_over():
                    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                    overlay.fill((0, 0, 0, 170))
                    screen.blit(overlay, (0, 0))

                    winner = game.get_winner()
                    if winner == 0:
                        msg = "Draw!"
                    else:
                        msg = f"Player {winner} Won!"

                    draw_centered_text(screen, msg, title_font, HEIGHT // 2 - 40, ACCENT_COLOR)
                    draw_centered_text(
                        screen,
                        "Select one option:",
                        small_font,
                        HEIGHT // 2,
                        TEXT_COLOR,
                    )

                    if not game_over_handled:
                        setup_endgame_buttons()

                    for b in end_buttons:
                        b.draw(screen, mouse_pos)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

def run_mirror(
    game_name: str,
    log_path: str = "mirror_log.txt",
    input_path: str = "input_log.txt",
    p1_human: bool = False,
    p2_human: bool = False,
):
    """
    Opens the UI as a mirror of the game running in play.py.
    - game_name: "gomoku" or "pente"
    - log_path: file where play.py writes the moves as 'row col'
    - input_path: file where this UI writes moves for human players
    - p1_human / p2_human: whether each player is human and plays via this UI.

    This does NOT control the game logic; play.py remains the motor.
    """
    pygame.init()
    pygame.display.set_caption("Lab IACD - Mirror View")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    title_font = get_chinese_font(70, bold=True)
    small_font = get_chinese_font(22)
    tiny_font = get_chinese_font(18)

    # background image (same as main)
    menu_bg = None
    try:
        img = pygame.image.load(MENU_BG_IMAGE_PATH)
        menu_bg = pygame.transform.smoothscale(img, (WIDTH, HEIGHT))
    except Exception:
        menu_bg = None

    # local game to mirror the moves
    if game_name == "gomoku":
        game = Gomoku(size=BOARD_SIZE)
    else:
        game = Pente(size=BOARD_SIZE)

    # how many move lines we've already applied
    applied_moves = 0

    # track end of game
    game_over = False
    winner = None

    # controlar input humano: só 1 jogada por vez, até o play.py a aceitar
    waiting_for_input = False
    last_moves_ackd = 0  # quantas jogadas já vimos no log

    running = True
    while running:
        dt = clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # clique para HUMANO jogar via interface
            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and waiting_for_input
                and not game_over
            ):
                pos = screen_to_board(event.pos)
                if pos is not None:
                    r, c = pos
                    # só permite jogar em casas vazias
                    if game.board[r, c] == 0:
                        # escreve a jogada no input_log.txt
                        try:
                            with open(input_path, "a", encoding="utf-8") as f:
                                f.write(f"{r} {c}\n")
                            waiting_for_input = False  # espera agora o play.py aplicar e registar no log
                        except Exception:
                            pass

        # === read log and apply new moves ===
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            lines = []

        start_idx = 0
        if lines and lines[0].startswith("GAME"):
            start_idx = 1

        move_lines = lines[start_idx:]

        # apply only the moves we haven't applied yet
        for line in move_lines[applied_moves:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                r, c = map(int, parts)
                game.do_move((r, c))
            except Exception:
                # ignore malformed or invalid moves in the log
                pass

        applied_moves = len(move_lines)

        # detectar novas jogadas confirmadas pelo play.py
        if applied_moves > last_moves_ackd:
            last_moves_ackd = applied_moves
            waiting_for_input = False  # uma jogada nova entrou; próxima vez que for humano, voltamos a aceitar input

        # check if game ended
        if not game_over and game.is_game_over():
            game_over = True
            winner = game.get_winner()

        # determinar se é a vez de um humano jogar (depois de aplicar tudo do log)
        if not game_over:
            current_player = game.current_player
            is_human_turn = (
                (current_player == 1 and p1_human)
                or (current_player == 2 and p2_human)
            )
        else:
            is_human_turn = False

        if is_human_turn and not game_over and not waiting_for_input:
            waiting_for_input = True

        # === draw ===
        if menu_bg is not None:
            screen.blit(menu_bg, (0, 0))
        else:
            screen.fill(BG_COLOR)

        # game title with shadow (same style as your PLAYING state)
        game_label = "GOMOKU" if isinstance(game, Gomoku) else "PENTE"
        shadow_surf = title_font.render(game_label, True, (0, 0, 0))
        shadow_rect = shadow_surf.get_rect(center=(WIDTH // 2 + 2, 55 + 2))
        screen.blit(shadow_surf, shadow_rect)
        draw_centered_text(screen, game_label, title_font, 55, ACCENT_COLOR)

        # info text
        info_text = "Mirror view"
        screen.blit(small_font.render(info_text, True, TEXT_COLOR), (20, 20))

        # draw board and last move highlight
        draw_board(screen, game)
        draw_last_move_ring(screen, game)

        # ghost stone quando é a vez do humano
        if waiting_for_input and not game_over and is_human_turn:
            current_player = game.current_player
            draw_ghost_stone(screen, game, current_player, mouse_pos)

        # if game is over, draw overlay + message
        if game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))  # semi-transparent dark layer
            screen.blit(overlay, (0, 0))

            if winner == 0:
                msg = "Draw!"
            elif winner == 1:
                msg = "Player 1 (Black) Won!"
            else:
                msg = "Player 2 (White) Won!"

            draw_centered_text(screen, msg, title_font, HEIGHT // 2 - 20, ACCENT_COLOR)
            draw_centered_text(
                screen,
                "Check terminal for details",
                small_font,
                HEIGHT // 2 + 25,
                TEXT_COLOR,
            )

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    # Usage:
    #   python interface_pygame.py
    #       -> normal menu & UI
    #
    #   python interface_pygame.py mirror
    #       -> mirror Gomoku by default, ambos bots
    #
    #   python interface_pygame.py mirror gomoku human bot
    #       -> P1 humano na UI, P2 bot
    #
    if len(sys.argv) >= 2 and sys.argv[1] == "mirror":
        game_name = "gomoku"
        if len(sys.argv) >= 3:
            game_name = sys.argv[2]

        p1_human = False
        p2_human = False
        if len(sys.argv) >= 4:
            p1_human = sys.argv[3].lower() == "human"
        if len(sys.argv) >= 5:
            p2_human = sys.argv[4].lower() == "human"

        run_mirror(
            game_name,
            log_path="mirror_log.txt",
            input_path="input_log.txt",
            p1_human=p1_human,
            p2_human=p2_human,
        )
    else:
        main()