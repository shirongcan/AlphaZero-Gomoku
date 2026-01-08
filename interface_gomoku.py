import pygame
import sys
import importlib
import math
import os

from games.gomoku import Gomoku

WIDTH, HEIGHT = 900, 920
FPS = 60

BG_COLOR = (18, 12, 24)
PANEL_COLOR = (45, 35, 70)
# 更暗的木纹/胡桃木基调
BOARD_BG_COLOR = (196, 165, 118)
GRID_COLOR = (78, 56, 34)
HIGHLIGHT_COLOR = (255, 230, 120)

BLACK_STONE = (25, 25, 30)
# 白棋降低亮度，偏暖的象牙白
WHITE_STONE = (224, 220, 210)

TEXT_COLOR = (235, 235, 235)
ACCENT_COLOR = (140, 110, 190)
BUTTON_HOVER = (75, 65, 110)
BUTTON_SELECTED = (140, 110, 190)

BOARD_SIZE = 15
BOARD_PIXEL_SIZE = 660
CELL_SIZE = BOARD_PIXEL_SIZE // BOARD_SIZE
BOARD_ORIGIN_X = (WIDTH - BOARD_PIXEL_SIZE) // 2
# 棋盘下移，避免遮挡顶部信息区
BOARD_ORIGIN_Y = 150

STATE_PLAYER_SELECT = "player_select"
STATE_MODEL_SELECT = "model_select"
STATE_PLAYING = "playing"

MENU_BG_IMAGE_PATH = "interface_menus/menu3.webp"


def list_model_files(models_dir="models"):
    """
    返回 models/ 下所有 .pt 文件（按修改时间从新到旧排序），元素为相对路径字符串。
    """
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


def truncate_to_width(font, text, max_width):
    """
    把文本截断到不超过 max_width（像素），末尾用 … 表示。
    """
    if font.size(text)[0] <= max_width:
        return text
    ell = "…"
    lo, hi = 0, len(text)
    best = ell
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[:mid] + ell
        if font.size(cand)[0] <= max_width:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def get_chinese_font(size, bold=False):
    """
    获取支持中文的字体。尝试多个中文字体，如果都失败则使用系统默认字体。
    """
    chinese_fonts = [
        "Microsoft YaHei",  # 微软雅黑
        "SimHei",  # 黑体
        "SimSun",  # 宋体
        "KaiTi",  # 楷体
        "FangSong",  # 仿宋
    ]

    for font_name in chinese_fonts:
        try:
            font = pygame.font.SysFont(font_name, size, bold=bold)
            test_surf = font.render("测试", True, (255, 255, 255))
            if test_surf.get_width() > 0:
                return font
        except Exception:
            continue

    try:
        return pygame.font.SysFont(None, size, bold=bold)
    except Exception:
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
        self.visible = True

    def draw(self, surface, mouse_pos):
        if not self.visible:
            return
        is_hover = self.rect.collidepoint(mouse_pos)

        draw_rect = self.rect.copy()
        if is_hover:
            draw_rect.inflate_ip(6, 4)

        shadow_rect = draw_rect.copy()
        shadow_rect.x += 3
        shadow_rect.y += 4
        pygame.draw.rect(surface, (0, 0, 0, 80), shadow_rect, border_radius=12)

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
        if not self.visible:
            return
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
    raise ValueError(f"在 {module_name} 中没有找到 Player 类")


def draw_centered_text(surface, text, font, y, color=TEXT_COLOR, pulse=0.0):
    text_surf = font.render(text, True, color)
    if pulse > 0:
        scale = 1.0 + 0.06 * pulse
        w, h = text_surf.get_size()
        text_surf = pygame.transform.smoothscale(text_surf, (int(w * scale), int(h * scale)))
    text_rect = text_surf.get_rect(center=(WIDTH // 2, y))
    surface.blit(text_surf, text_rect)


def cell_center(row, col):
    cx = BOARD_ORIGIN_X + CELL_SIZE // 2 + col * CELL_SIZE
    cy = BOARD_ORIGIN_Y + CELL_SIZE // 2 + row * CELL_SIZE
    return cx, cy


def draw_board(surface, game):
    board_rect = pygame.Rect(BOARD_ORIGIN_X, BOARD_ORIGIN_Y, BOARD_PIXEL_SIZE, BOARD_PIXEL_SIZE)

    shadow_rect = board_rect.copy()
    shadow_rect.topleft = (shadow_rect.left + 4, shadow_rect.top + 6)
    pygame.draw.rect(surface, (10, 8, 18), shadow_rect, border_radius=16)

    pygame.draw.rect(surface, BOARD_BG_COLOR, board_rect, border_radius=16)
    pygame.draw.rect(surface, GRID_COLOR, board_rect, 3, border_radius=16)

    for i in range(BOARD_SIZE):
        start = (BOARD_ORIGIN_X + CELL_SIZE // 2, BOARD_ORIGIN_Y + CELL_SIZE // 2 + i * CELL_SIZE)
        end = (
            BOARD_ORIGIN_X + BOARD_PIXEL_SIZE - CELL_SIZE // 2,
            BOARD_ORIGIN_Y + CELL_SIZE // 2 + i * CELL_SIZE,
        )
        pygame.draw.line(surface, GRID_COLOR, start, end, 1)

        start = (BOARD_ORIGIN_X + CELL_SIZE // 2 + i * CELL_SIZE, BOARD_ORIGIN_Y + CELL_SIZE // 2)
        end = (
            BOARD_ORIGIN_X + CELL_SIZE // 2 + i * CELL_SIZE,
            BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE - CELL_SIZE // 2,
        )
        pygame.draw.line(surface, GRID_COLOR, start, end, 1)

    # 天元（星位点）：在中心位置画一个小圆点
    if BOARD_SIZE % 2 == 1:
        center = BOARD_SIZE // 2
        cx, cy = cell_center(center, center)
        star_color = (60, 44, 26)  # 深棕色，更像木纹上的星位
        star_radius = max(3, CELL_SIZE // 8)
        pygame.draw.circle(surface, star_color, (cx, cy), star_radius)

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


def screen_to_board(pos):
    x, y = pos
    if not (
        BOARD_ORIGIN_X <= x < BOARD_ORIGIN_X + BOARD_PIXEL_SIZE
        and BOARD_ORIGIN_Y <= y < BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE
    ):
        return None

    col = (x - BOARD_ORIGIN_X) // CELL_SIZE
    row = (y - BOARD_ORIGIN_Y) // CELL_SIZE
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return (row, col)
    return None


def build_replay_game(original_game, upto_moves):
    g = Gomoku(size=BOARD_SIZE)
    history = getattr(original_game, "move_history", [])
    upto_moves = min(upto_moves, len(history))
    for i in range(upto_moves):
        g.do_move(history[i])
    return g


def get_player_label(choice_key, idx):
    if choice_key == "human":
        return f"玩家{idx}: 人类"
    if choice_key == "mcts":
        return f"玩家{idx}: MCTS"
    if choice_key == "alpha":
        return f"玩家{idx}: AlphaZero"
    return f"玩家{idx}"


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


def main():
    pygame.init()
    pygame.display.set_caption("五子棋 - 对弈")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    title_font = get_chinese_font(70, bold=True)
    menu_font = get_chinese_font(28, bold=True)
    small_font = get_chinese_font(22)
    tiny_font = get_chinese_font(18)

    menu_bg = None
    try:
        img = pygame.image.load(MENU_BG_IMAGE_PATH)
        menu_bg = pygame.transform.smoothscale(img, (WIDTH, HEIGHT))
    except Exception:
        menu_bg = None

    state = STATE_PLAYER_SELECT

    selected_p1 = None
    selected_p2 = None
    selected_model_p1 = None
    selected_model_p2 = None

    game = None
    players = {}
    turn_number = 0

    buttons = []
    model_buttons = []
    model_nav_buttons = []
    end_buttons = []
    replay_buttons = []
    play_buttons = []

    game_over_handled = False
    replay_index = None
    replay_game = None
    model_scroll = 0
    MODEL_VISIBLE_COUNT = 7

    def reset_buttons():
        nonlocal buttons
        buttons = []

    def reset_model_buttons():
        nonlocal model_buttons, model_nav_buttons
        model_buttons = []
        model_nav_buttons = []

    def reset_end_buttons():
        nonlocal end_buttons, game_over_handled
        end_buttons = []
        game_over_handled = False

    def reset_replay():
        nonlocal replay_index, replay_game
        replay_index = None
        replay_game = None

    def reset_play_buttons():
        nonlocal play_buttons
        play_buttons = []

    def set_player1(choice_key):
        nonlocal selected_p1
        selected_p1 = choice_key
        for b in buttons:
            if getattr(b, "column", None) == 1:
                b.selected = b.data == choice_key

    def set_player2(choice_key):
        nonlocal selected_p2
        selected_p2 = choice_key
        for b in buttons:
            if getattr(b, "column", None) == 2:
                b.selected = b.data == choice_key

    def create_player(choice_key, model_path=None):
        if choice_key == "human":
            return HumanGUIPlayer()
        if choice_key == "mcts":
            return load_bot_player("player_mcts", "gomoku", BOARD_SIZE)
        if choice_key == "alpha":
            # 不修改原 player_alpha.py：这里改为使用新实现，并注入 model_path
            from players.player_alpha_gomoku import Player as AlphaGomokuPlayer

            return AlphaGomokuPlayer(rules="gomoku", board_size=BOARD_SIZE, model_path=model_path)
        raise ValueError("未知玩家类型")

    def setup_replay_buttons():
        nonlocal replay_buttons
        replay_buttons = []
        y = BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE // 2 - 20
        left_rect = (BOARD_ORIGIN_X - 70, y, 50, 40)
        right_rect = (BOARD_ORIGIN_X + BOARD_PIXEL_SIZE + 20, y, 50, 40)
        replay_buttons.append(Button(left_rect, "<", menu_font, lambda _: step_replay(-1)))
        replay_buttons.append(Button(right_rect, ">", menu_font, lambda _: step_replay(1)))

    def setup_play_buttons():
        nonlocal play_buttons
        reset_play_buttons()
        btn_w, btn_h = 120, 34
        margin = 20
        x = margin
        y = HEIGHT - btn_h - margin

        play_buttons.append(
            Button(
                rect=(x, y, btn_w, btn_h),
                text="返回菜单",
                font=tiny_font,
                callback=back_to_menu,
            )
        )

        # 底部第二按钮：
        # - 终局时：复盘
        # - 复盘时：再来一盘（更符合用户预期）
        b2 = Button(
            rect=(x + btn_w + 12, y, btn_w, btn_h),
            text="复盘",
            font=tiny_font,
            callback=secondary_bottom_action,
        )
        b2.visible = False  # 只在终局/复盘时显示
        play_buttons.append(b2)

    def start_new_match():
        nonlocal game, players, turn_number, state
        reset_end_buttons()
        reset_replay()

        game_local = Gomoku(size=BOARD_SIZE)
        p1 = create_player(selected_p1, selected_model_p1)
        p2 = create_player(selected_p2, selected_model_p2)

        game = game_local
        players = {1: p1, 2: p2}
        turn_number = 0
        state = STATE_PLAYING
        setup_replay_buttons()
        setup_play_buttons()

    def go_to_model_select():
        nonlocal state, model_scroll
        state = STATE_MODEL_SELECT
        reset_buttons()
        reset_model_buttons()
        model_scroll = 0
        setup_model_select_buttons()

    def start_game_if_ready(_):
        if selected_p1 is None or selected_p2 is None:
            return
        if selected_p1 == "alpha" or selected_p2 == "alpha":
            go_to_model_select()
        else:
            start_new_match()
            reset_buttons()

    def back_to_menu(_=None):
        nonlocal state, selected_p1, selected_p2, selected_model_p1, selected_model_p2, game, model_scroll
        state = STATE_PLAYER_SELECT
        selected_p1 = None
        selected_p2 = None
        selected_model_p1 = None
        selected_model_p2 = None
        game = None
        model_scroll = 0
        reset_buttons()
        reset_model_buttons()
        reset_end_buttons()
        reset_replay()
        reset_play_buttons()
        setup_player_select_buttons()

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

    def toggle_replay_mode(_=None):
        """
        - 当游戏已结束且未在复盘：进入复盘（从最后一手开始）
        - 当正在复盘：退出复盘，回到终局界面
        """
        nonlocal replay_index, replay_game
        if game is None:
            return
        if replay_index is None:
            if not game.is_game_over():
                return
            step_replay(-1)
        else:
            replay_index = None
            replay_game = None

    def secondary_bottom_action(_=None):
        """
        底部第二按钮的语义：
        - 未进入复盘：进入复盘
        - 已进入复盘：再来一盘（开新局）
        """
        if replay_index is None:
            toggle_replay_mode()
        else:
            start_new_match()

    def enter_replay_from_endgame(_=None):
        # 专供终局按钮使用：进入复盘
        toggle_replay_mode()

    def back_to_player_select(_=None):
        nonlocal state, model_scroll
        state = STATE_PLAYER_SELECT
        model_scroll = 0
        reset_model_buttons()
        setup_player_select_buttons()

    def set_model_for_player(data):
        nonlocal selected_model_p1, selected_model_p2
        which, path = data
        if which == 1:
            selected_model_p1 = path
        else:
            selected_model_p2 = path
        # 更新选中态
        for b in model_buttons:
            if getattr(b, "column", None) == which:
                b.selected = b.data == (which, path)

    def clamp_model_scroll(total):
        nonlocal model_scroll
        max_scroll = max(0, total - MODEL_VISIBLE_COUNT)
        if model_scroll < 0:
            model_scroll = 0
        if model_scroll > max_scroll:
            model_scroll = max_scroll

    def scroll_models(delta):
        nonlocal model_scroll
        models = list_model_files("models")
        model_scroll += delta
        clamp_model_scroll(len(models))
        setup_model_select_buttons()

    def ensure_default_model_selection(models):
        nonlocal selected_model_p1, selected_model_p2
        if not models:
            return
        if selected_p1 == "alpha" and selected_model_p1 is None:
            selected_model_p1 = models[0]
        if selected_p2 == "alpha" and selected_model_p2 is None:
            selected_model_p2 = models[0]

    def confirm_models_and_start(_=None):
        models = list_model_files("models")
        ensure_default_model_selection(models)

        # 必须为 AlphaZero 选择模型
        if selected_p1 == "alpha" and not selected_model_p1:
            return
        if selected_p2 == "alpha" and not selected_model_p2:
            return

        start_new_match()
        reset_model_buttons()

    def setup_model_select_buttons():
        """
        模型选择页：为每个 AlphaZero 玩家提供 models/*.pt 列表。
        鼠标滚轮可上下滚动（同时影响两列）。
        """
        nonlocal model_buttons, model_nav_buttons
        reset_model_buttons()

        models = list_model_files("models")
        ensure_default_model_selection(models)
        clamp_model_scroll(len(models))

        col_w = 320
        col_gap = 60
        total_w = col_w * 2 + col_gap
        base_x = (WIDTH - total_w) // 2
        col1_x = base_x
        col2_x = base_x + col_w + col_gap

        # 模型按钮列表的起始位置（需留出标题/提示/玩家信息区）
        y0 = 330
        btn_h = 42
        spacing = 8

        visible = models[model_scroll : model_scroll + MODEL_VISIBLE_COUNT]

        # 为了避免文字溢出，做截断
        max_text_w = col_w - 24

        if selected_p1 == "alpha":
            for i, path in enumerate(visible):
                name = os.path.basename(path)
                text = truncate_to_width(tiny_font, name, max_text_w)
                rect = (col1_x, y0 + i * (btn_h + spacing), col_w, btn_h)
                b = Button(rect, text, tiny_font, set_model_for_player, data=(1, path))
                b.column = 1
                b.selected = (selected_model_p1 == path)
                model_buttons.append(b)

        if selected_p2 == "alpha":
            for i, path in enumerate(visible):
                name = os.path.basename(path)
                text = truncate_to_width(tiny_font, name, max_text_w)
                rect = (col2_x, y0 + i * (btn_h + spacing), col_w, btn_h)
                b = Button(rect, text, tiny_font, set_model_for_player, data=(2, path))
                b.column = 2
                b.selected = (selected_model_p2 == path)
                model_buttons.append(b)

        # 导航按钮：滚动、返回、开始
        nav_y = HEIGHT - 120
        model_nav_buttons.append(
            Button((WIDTH // 2 - 300, nav_y, 180, 52), "返回玩家选择", menu_font, back_to_player_select)
        )
        model_nav_buttons.append(
            Button((WIDTH // 2 + 120, nav_y, 180, 52), "开始对弈", menu_font, confirm_models_and_start)
        )

        # 上/下滚动按钮（给没滚轮的人）
        model_nav_buttons.append(Button((WIDTH // 2 - 90, nav_y, 80, 52), "上移", menu_font, lambda _: scroll_models(-1)))
        model_nav_buttons.append(Button((WIDTH // 2 + 10, nav_y, 80, 52), "下移", menu_font, lambda _: scroll_models(1)))

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
            ("人类", "human"),
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
        buttons.append(Button(start_rect, "开始对弈", menu_font, start_game_if_ready))

    def setup_endgame_buttons():
        nonlocal end_buttons, game_over_handled
        end_buttons = []
        game_over_handled = True

        btn_w, btn_h = 190, 55
        gap = 26
        total_w = btn_w * 3 + gap * 2
        start_x = (WIDTH - total_w) // 2
        # 终局按钮放到棋盘下方，避免遮挡棋盘
        y = BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE + 14

        end_buttons.append(
            Button(
                rect=(start_x, y, btn_w, btn_h),
                text="再来一盘",
                font=menu_font,
                callback=lambda _: start_new_match(),
            )
        )
        end_buttons.append(
            Button(
                rect=(start_x + btn_w + gap, y, btn_w, btn_h),
                text="复盘",
                font=menu_font,
                callback=enter_replay_from_endgame,
            )
        )
        end_buttons.append(
            Button(
                rect=(start_x + (btn_w + gap) * 2, y, btn_w, btn_h),
                text="返回菜单",
                font=menu_font,
                callback=back_to_menu,
            )
        )

    setup_player_select_buttons()

    running = True
    while running:
        clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()
        t = pygame.time.get_ticks() / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if state == STATE_PLAYER_SELECT:
                for b in buttons:
                    b.handle_event(event)

            elif state == STATE_MODEL_SELECT:
                for b in model_buttons:
                    b.handle_event(event)
                for b in model_nav_buttons:
                    b.handle_event(event)
                if event.type == pygame.MOUSEWHEEL:
                    # pygame: y>0 向上滚，y<0 向下滚
                    scroll_models(-event.y)

            elif state == STATE_PLAYING:
                for b in replay_buttons:
                    b.handle_event(event)
                for b in play_buttons:
                    b.handle_event(event)

                if game and (not game.is_game_over()) and replay_index is None:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        move = screen_to_board(event.pos)
                        if move is not None:
                            current_player = game.current_player
                            if isinstance(players.get(current_player), HumanGUIPlayer):
                                r, c = move
                                if game.board[r, c] == 0:
                                    players[current_player].set_click(move)
                elif game and game.is_game_over() and replay_index is None:
                    for b in end_buttons:
                        b.handle_event(event)

        if menu_bg is not None:
            screen.blit(menu_bg, (0, 0))
        else:
            screen.fill(BG_COLOR)

        if state == STATE_PLAYER_SELECT:
            pulse = (math.sin(t * 2.0) + 1) / 2
            draw_centered_text(screen, "五子棋", title_font, 120, ACCENT_COLOR, pulse=pulse)
            draw_centered_text(screen, "选择双方玩家", menu_font, 180)

            screen.blit(small_font.render("玩家1（黑）", True, TEXT_COLOR), (WIDTH // 2 - 260, 200))
            screen.blit(small_font.render("玩家2（白）", True, TEXT_COLOR), (WIDTH // 2 + 40, 200))

            for b in buttons:
                b.draw(screen, mouse_pos)

        elif state == STATE_MODEL_SELECT:
            # 顶部信息区：拉开间距，避免与下面按钮区域重叠
            draw_centered_text(screen, "选择 AlphaZero 模型", title_font, 105, ACCENT_COLOR)
            draw_centered_text(
                screen,
                "提示：仅当玩家选择 AlphaZero 时需要选择模型（滚轮可滚动列表）",
                small_font,
                175,
                TEXT_COLOR,
            )

            # 标题
            col_w = 320
            col_gap = 60
            total_w = col_w * 2 + col_gap
            base_x = (WIDTH - total_w) // 2
            col1_x = base_x
            col2_x = base_x + col_w + col_gap

            screen.blit(small_font.render("玩家1（黑）", True, TEXT_COLOR), (col1_x, 235))
            screen.blit(small_font.render("玩家2（白）", True, TEXT_COLOR), (col2_x, 235))

            # 当前选择展示
            if selected_p1 == "alpha":
                s = os.path.basename(selected_model_p1) if selected_model_p1 else "未选择"
            else:
                s = "无需选择"
            screen.blit(tiny_font.render(f"模型: {s}", True, TEXT_COLOR), (col1_x, 265))

            if selected_p2 == "alpha":
                s = os.path.basename(selected_model_p2) if selected_model_p2 else "未选择"
            else:
                s = "无需选择"
            screen.blit(tiny_font.render(f"模型: {s}", True, TEXT_COLOR), (col2_x, 265))

            # 如果某一侧不是 AlphaZero，提示
            if selected_p1 != "alpha":
                screen.blit(tiny_font.render("（该玩家不是 AlphaZero）", True, TEXT_COLOR), (col1_x, 292))
            if selected_p2 != "alpha":
                screen.blit(tiny_font.render("（该玩家不是 AlphaZero）", True, TEXT_COLOR), (col2_x, 292))

            for b in model_buttons:
                b.draw(screen, mouse_pos)
            for b in model_nav_buttons:
                b.draw(screen, mouse_pos)

        elif state == STATE_PLAYING and game is not None:
            active_game = replay_game if replay_index is not None else game

            shadow_surf = title_font.render("GOMOKU", True, (0, 0, 0))
            shadow_rect = shadow_surf.get_rect(center=(WIDTH // 2 + 2, 55 + 2))
            screen.blit(shadow_surf, shadow_rect)
            draw_centered_text(screen, "GOMOKU", title_font, 55, ACCENT_COLOR)

            if replay_index is None:
                if not game.is_game_over():
                    turn_text = f"当前回合：玩家 {game.current_player}"
                else:
                    winner = game.get_winner()
                    turn_text = "平局" if winner == 0 else f"玩家 {winner} 获胜！"
            else:
                history = getattr(game, "move_history", [])
                turn_text = f"复盘：第 {replay_index + 1}/{len(history)} 手"

            screen.blit(small_font.render(turn_text, True, TEXT_COLOR), (20, 20))

            p1_label = get_player_label(selected_p1, 1)
            p2_label = get_player_label(selected_p2, 2)
            screen.blit(tiny_font.render(p1_label, True, TEXT_COLOR), (20, 50))
            screen.blit(tiny_font.render(p2_label, True, TEXT_COLOR), (20, 70))

            # 显示 AlphaZero 模型（如果选了）
            if selected_p1 == "alpha":
                m1 = os.path.basename(selected_model_p1) if selected_model_p1 else "未选择"
                screen.blit(tiny_font.render(f"P1模型: {m1}", True, TEXT_COLOR), (20, 92))
            if selected_p2 == "alpha":
                m2 = os.path.basename(selected_model_p2) if selected_model_p2 else "未选择"
                screen.blit(tiny_font.render(f"P2模型: {m2}", True, TEXT_COLOR), (20, 112))

            draw_board(screen, active_game)
            draw_last_move_ring(screen, active_game)

            for b in replay_buttons:
                # 复盘左右箭头只在进入复盘后显示
                b.visible = replay_index is not None
                b.draw(screen, mouse_pos)
            # 底部按钮：根据是否在复盘中更新文案
            if len(play_buttons) >= 2:
                play_buttons[1].text = "再来一盘" if replay_index is not None else "复盘"
                # 对局中隐藏“复盘”，终局/复盘中显示
                play_buttons[1].visible = (game.is_game_over() or replay_index is not None)
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
                # 只有“未进入复盘”的终局状态才画遮罩 + 终局按钮
                if game.is_game_over() and replay_index is None:
                    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                    overlay.fill((0, 0, 0, 170))
                    screen.blit(overlay, (0, 0))

                    winner = game.get_winner()
                    msg = "平局" if winner == 0 else f"玩家 {winner} 获胜！"
                    draw_centered_text(screen, msg, title_font, HEIGHT // 2 - 40, ACCENT_COLOR)
                    draw_centered_text(screen, "请选择：", small_font, HEIGHT // 2, TEXT_COLOR)

                    if not game_over_handled:
                        setup_endgame_buttons()

                    for b in end_buttons:
                        b.draw(screen, mouse_pos)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()


