import math

import pygame


class PygameVisualizer:
    """
    Pygame visualization of the Escape Room environment.

    The visualizer renders the grid world with a dark "dungeon" aesthetic:
    stone floor tiles, brick walls, a drawn golden key, rune-style puzzle
    tiles, a wooden door with a padlock, a glowing exit portal and a HUD
    panel with live episode status.

    Cell symbols (provided by the environment):
        A = Agent
        K = Key
        M/L/P = Puzzles
        D = Closed Door
        O = Open Door
        E = Exit
        # = Wall
    """

    COLORS = {
        "background": (18, 20, 28),
        "floor_a": (40, 44, 58),
        "floor_b": (35, 39, 52),
        "floor_edge": (26, 29, 39),
        "wall_brick": (62, 55, 50),
        "wall_brick_hi": (74, 66, 60),
        "wall_mortar": (30, 27, 25),
        "agent": (64, 200, 255),
        "agent_dark": (24, 120, 170),
        "agent_glow": (64, 200, 255),
        "key": (255, 209, 74),
        "key_dark": (170, 130, 20),
        "puzzle": (155, 110, 245),
        "puzzle_dark": (95, 60, 170),
        "puzzle_glow": (155, 110, 245),
        "door_wood": (128, 84, 46),
        "door_wood_dark": (96, 62, 34),
        "door_frame": (58, 50, 44),
        "door_open": (86, 200, 130),
        "lock": (210, 214, 224),
        "exit": (72, 214, 130),
        "exit_dark": (24, 120, 70),
        "text": (232, 235, 242),
        "text_dim": (150, 156, 172),
        "panel": (24, 26, 35),
        "panel_edge": (48, 52, 68),
        "chip_on": (46, 120, 80),
        "chip_off": (70, 52, 56),
        "chip_text_on": (190, 245, 210),
        "chip_text_off": (235, 190, 195),
        "bar_bg": (42, 46, 60),
        "bar_fill": (64, 160, 255),
        "reward_pos": (120, 230, 150),
        "reward_neg": (245, 130, 130),
    }

    # Auto-scaling keeps large grids inside a reasonable window:
    # cell_size = min(MAX_CELL_SIZE, MAX_BOARD_PIXELS // max(rows, cols)).
    MAX_CELL_SIZE = 80
    MAX_BOARD_PIXELS = 720

    def __init__(self, env, cell_size=None, panel_height=140):
        pygame.init()
        pygame.font.init()

        if cell_size is None:
            cell_size = min(
                self.MAX_CELL_SIZE,
                self.MAX_BOARD_PIXELS // max(env.rows, env.cols),
            )

        self.env = env
        self.cell_size = cell_size
        self.panel_height = panel_height

        self.width = env.cols * cell_size
        self.height = env.rows * cell_size + panel_height

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Escape Room ML Agent")

        font_name = "dejavusans,verdana,arial"
        self.font = pygame.font.SysFont(font_name, 20, bold=True)
        self.small_font = pygame.font.SysFont(font_name, 15)
        self.chip_font = pygame.font.SysFont(font_name, 14, bold=True)
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    # ------------------------------------------------------------------
    # Low-level drawing helpers
    # ------------------------------------------------------------------
    def _cell_rect(self, row, col):
        return pygame.Rect(
            col * self.cell_size,
            row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def _draw_glow(self, center, radius, color, strength=70):
        """Soft radial glow drawn on an alpha surface."""
        size = radius * 2
        glow = pygame.Surface((size, size), pygame.SRCALPHA)
        for i in range(radius, 0, -2):
            alpha = int(strength * (1 - i / radius))
            pygame.draw.circle(glow, (*color, alpha), (radius, radius), i)
        self.screen.blit(glow, (center[0] - radius, center[1] - radius))

    def _draw_floor(self, row, col):
        rect = self._cell_rect(row, col)
        color = self.COLORS["floor_a"] if (row + col) % 2 == 0 else self.COLORS["floor_b"]
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, self.COLORS["floor_edge"], rect, 1)

    def _draw_wall(self, row, col):
        rect = self._cell_rect(row, col)
        pygame.draw.rect(self.screen, self.COLORS["wall_mortar"], rect)

        brick_h = self.cell_size // 4
        brick_w = self.cell_size // 2
        for band in range(4):
            y = rect.y + band * brick_h
            offset = 0 if band % 2 == 0 else -brick_w // 2
            x = rect.x + offset
            while x < rect.right:
                brick = pygame.Rect(x + 2, y + 2, brick_w - 3, brick_h - 3)
                brick = brick.clip(rect.inflate(-2, -2))
                if brick.width > 2 and brick.height > 2:
                    pygame.draw.rect(self.screen, self.COLORS["wall_brick"], brick, border_radius=2)
                    highlight = pygame.Rect(brick.x, brick.y, brick.width, 2)
                    pygame.draw.rect(self.screen, self.COLORS["wall_brick_hi"], highlight)
                x += brick_w

    def _draw_agent(self, rect):
        center = rect.center
        radius = self.cell_size // 3

        self._draw_glow(center, int(radius * 1.7), self.COLORS["agent_glow"], strength=55)

        pygame.draw.circle(self.screen, self.COLORS["agent_dark"], center, radius)
        pygame.draw.circle(self.screen, self.COLORS["agent"], center, radius - 3)

        # visor / eyes
        eye_dy = radius // 4
        eye_dx = radius // 3
        eye_r = max(3, radius // 6)
        for dx in (-eye_dx, eye_dx):
            pygame.draw.circle(
                self.screen,
                (12, 30, 44),
                (center[0] + dx, center[1] - eye_dy),
                eye_r,
            )

        # small antenna
        pygame.draw.line(
            self.screen,
            self.COLORS["agent_dark"],
            (center[0], center[1] - radius + 2),
            (center[0], center[1] - radius - 7),
            3,
        )
        pygame.draw.circle(
            self.screen,
            self.COLORS["key"],
            (center[0], center[1] - radius - 9),
            4,
        )

    def _draw_key(self, rect):
        center = rect.center
        scale = self.cell_size / 80.0

        self._draw_glow(center, int(26 * scale) * 2 // 2 + int(14 * scale), self.COLORS["key"], strength=45)

        bow_r = int(11 * scale)
        bow_center = (center[0] - int(14 * scale), center[1] - int(10 * scale))
        shaft_end = (center[0] + int(18 * scale), center[1] + int(14 * scale))

        # shaft
        pygame.draw.line(self.screen, self.COLORS["key_dark"], bow_center, shaft_end, int(7 * scale))
        pygame.draw.line(self.screen, self.COLORS["key"], bow_center, shaft_end, int(4 * scale))

        # teeth
        for i, t in enumerate((0.72, 0.9)):
            tx = bow_center[0] + (shaft_end[0] - bow_center[0]) * t
            ty = bow_center[1] + (shaft_end[1] - bow_center[1]) * t
            pygame.draw.line(
                self.screen,
                self.COLORS["key"],
                (tx, ty),
                (tx + int(7 * scale), ty - int(7 * scale)),
                int(4 * scale),
            )

        # bow (ring)
        pygame.draw.circle(self.screen, self.COLORS["key_dark"], bow_center, bow_r)
        pygame.draw.circle(self.screen, self.COLORS["key"], bow_center, bow_r - 2)
        pygame.draw.circle(self.screen, self.COLORS["floor_a"], bow_center, max(3, bow_r - 6))

    def _draw_puzzle(self, rect, symbol):
        tile = rect.inflate(-int(self.cell_size * 0.3), -int(self.cell_size * 0.3))

        self._draw_glow(rect.center, int(self.cell_size * 0.45), self.COLORS["puzzle_glow"], strength=40)

        pygame.draw.rect(self.screen, self.COLORS["puzzle_dark"], tile, border_radius=10)
        pygame.draw.rect(self.screen, self.COLORS["puzzle"], tile.inflate(-6, -6), border_radius=8)

        surface = self.font.render(symbol, True, (25, 12, 50))
        self.screen.blit(surface, surface.get_rect(center=tile.center))

    def _draw_door(self, rect, is_open):
        frame = rect.inflate(-int(self.cell_size * 0.18), -int(self.cell_size * 0.10))
        pygame.draw.rect(self.screen, self.COLORS["door_frame"], frame, border_radius=6)

        door = frame.inflate(-8, -8)

        if is_open:
            # dark opening with a green glow: the way forward
            pygame.draw.rect(self.screen, (12, 14, 20), door, border_radius=4)
            self._draw_glow(rect.center, int(self.cell_size * 0.4), self.COLORS["door_open"], strength=50)
            pygame.draw.rect(self.screen, self.COLORS["door_open"], door, 2, border_radius=4)
        else:
            pygame.draw.rect(self.screen, self.COLORS["door_wood"], door, border_radius=4)
            # wooden planks
            plank_w = max(6, door.width // 3)
            for x in range(door.x + plank_w, door.right, plank_w):
                pygame.draw.line(
                    self.screen,
                    self.COLORS["door_wood_dark"],
                    (x, door.y + 2),
                    (x, door.bottom - 2),
                    2,
                )
            # padlock
            lock_center = (door.centerx, door.centery + door.height // 6)
            body = pygame.Rect(0, 0, int(self.cell_size * 0.22), int(self.cell_size * 0.18))
            body.center = lock_center
            shackle_r = body.width // 2 - 2
            pygame.draw.circle(
                self.screen,
                self.COLORS["lock"],
                (body.centerx, body.y),
                shackle_r,
                3,
            )
            pygame.draw.rect(self.screen, self.COLORS["lock"], body, border_radius=4)
            pygame.draw.circle(self.screen, self.COLORS["door_wood_dark"], body.center, 3)

    def _draw_exit(self, rect):
        self._draw_glow(rect.center, int(self.cell_size * 0.55), self.COLORS["exit"], strength=60)

        portal = rect.inflate(-int(self.cell_size * 0.25), -int(self.cell_size * 0.25))
        pygame.draw.rect(self.screen, self.COLORS["exit_dark"], portal, border_radius=12)
        pygame.draw.rect(self.screen, self.COLORS["exit"], portal, 3, border_radius=12)

        surface = self.chip_font.render("EXIT", True, self.COLORS["chip_text_on"])
        self.screen.blit(surface, surface.get_rect(center=portal.center))

    # ------------------------------------------------------------------
    # Cell dispatch
    # ------------------------------------------------------------------
    def draw_cell(self, row, col, symbol):
        rect = self._cell_rect(row, col)

        if symbol == "#":
            self._draw_wall(row, col)
            return

        self._draw_floor(row, col)

        if symbol == "A":
            self._draw_agent(rect)
        elif symbol == "K":
            self._draw_key(rect)
        elif symbol in ("M", "L", "P"):
            self._draw_puzzle(rect, symbol)
        elif symbol == "D":
            self._draw_door(rect, is_open=False)
        elif symbol == "O":
            self._draw_door(rect, is_open=True)
        elif symbol == "E":
            self._draw_exit(rect)

    # ------------------------------------------------------------------
    # HUD panel
    # ------------------------------------------------------------------
    def _draw_chip(self, x, y, label, active):
        color = self.COLORS["chip_on"] if active else self.COLORS["chip_off"]
        text_color = self.COLORS["chip_text_on"] if active else self.COLORS["chip_text_off"]

        surface = self.chip_font.render(label, True, text_color)
        pad_x, pad_y = 10, 5
        chip = pygame.Rect(x, y, surface.get_width() + pad_x * 2, surface.get_height() + pad_y * 2)
        pygame.draw.rect(self.screen, color, chip, border_radius=chip.height // 2)
        self.screen.blit(surface, (chip.x + pad_x, chip.y + pad_y))
        return chip.right + 8

    def draw_panel(self, action_text="", reward=0, message=""):
        panel_y = self.env.rows * self.cell_size
        panel_rect = pygame.Rect(0, panel_y, self.width, self.panel_height)

        pygame.draw.rect(self.screen, self.COLORS["panel"], panel_rect)
        pygame.draw.line(
            self.screen,
            self.COLORS["panel_edge"],
            (0, panel_y),
            (self.width, panel_y),
            2,
        )

        margin = 12
        y = panel_y + 10

        # --- status chips ---
        solved_count = sum(1 for puzzle in self.env.puzzles if puzzle.solved)
        total_puzzles = len(self.env.puzzles)

        x = margin
        x = self._draw_chip(x, y, "KEY " + ("\u2713" if self.env.has_key else "\u2717"), self.env.has_key)
        x = self._draw_chip(x, y, "DOOR " + ("OPEN" if self.env.door_open else "LOCKED"), self.env.door_open)
        x = self._draw_chip(
            x,
            y,
            f"PUZZLES {solved_count}/{total_puzzles}",
            solved_count == total_puzzles and total_puzzles > 0,
        )

        # --- steps progress bar ---
        y += 32
        bar_h = 8
        bar = pygame.Rect(margin, y + 4, self.width - margin * 2 - 130, bar_h)
        pygame.draw.rect(self.screen, self.COLORS["bar_bg"], bar, border_radius=bar_h // 2)
        progress = min(1.0, self.env.steps / max(1, self.env.max_steps))
        if progress > 0:
            fill = pygame.Rect(bar.x, bar.y, max(bar_h, int(bar.width * progress)), bar_h)
            pygame.draw.rect(self.screen, self.COLORS["bar_fill"], fill, border_radius=bar_h // 2)

        steps_surface = self.small_font.render(
            f"Step {self.env.steps}/{self.env.max_steps}", True, self.COLORS["text_dim"]
        )
        self.screen.blit(steps_surface, (bar.right + 10, y))

        # --- action / reward line ---
        y += 24
        reward_color = self.COLORS["reward_pos"] if reward >= 0 else self.COLORS["reward_neg"]
        action_surface = self.small_font.render(f"Action: {action_text}", True, self.COLORS["text"])
        reward_surface = self.small_font.render(f"Reward: {reward:+}", True, reward_color)
        self.screen.blit(action_surface, (margin, y))
        self.screen.blit(reward_surface, (margin + action_surface.get_width() + 20, y))

        # --- message line ---
        y += 22
        if message:
            message_surface = self.small_font.render(message, True, self.COLORS["text_dim"])
            self.screen.blit(message_surface, (margin, y))

    # ------------------------------------------------------------------
    # Frame drawing
    # ------------------------------------------------------------------
    def draw(self, action_text="", reward=0, message=""):
        self.handle_events()
        self.screen.fill(self.COLORS["background"])

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                symbol = self.env.get_cell_symbol((row, col))
                self.draw_cell(row, col, symbol)

        self.draw_panel(action_text=action_text, reward=reward, message=message)

        pygame.display.flip()
        self.clock.tick(30)

    def get_frame(self):
        """Return the current screen as an (H, W, 3) uint8 array,
        e.g. for assembling GIF recordings."""
        import numpy as np

        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    def wait(self, milliseconds):
        pygame.time.delay(milliseconds)

    def close(self):
        pygame.quit()
