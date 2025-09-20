# pause_menu.py
import pygame
from settings import *
from button import Button

class PauseMenu:
    def __init__(self, game):
        self.game = game
        overlay_width = 400
        overlay_height = 300
        overlay_x = (WIDTH - overlay_width) // 2
        overlay_y = (HEIGHT - overlay_height) // 2
        self.overlay_rect = pygame.Rect(overlay_x, overlay_y, overlay_width, overlay_height)

        # Tạo các button bên trong menu
        btn_x = self.overlay_rect.centerx - BUTTON_WIDTH // 2
        self.resume_button = Button(btn_x, self.overlay_rect.y + 50, BUTTON_WIDTH, BUTTON_HEIGHT, text="Tiếp tục")
        self.home_button = Button(btn_x, self.overlay_rect.y + 150, BUTTON_WIDTH, BUTTON_HEIGHT, text="Về Trang chủ")

    def handle_input(self, cursor_pos, is_clicking):
        """Xử lý input và trả về hành động cần thực hiện."""
        self.resume_button.check_hover(cursor_pos)
        self.home_button.check_hover(cursor_pos)
        
        if is_clicking:
            if self.resume_button.is_hovered:
                return "resume"
            if self.home_button.is_hovered:
                return "go_home"
        return None

    def draw(self, surface):
        # Vẽ một lớp phủ bán trong suốt
        overlay_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay_surf.fill((0, 0, 0, 180)) # Màu đen, gần trong suốt
        surface.blit(overlay_surf, (0, 0))
        
        # Vẽ khung menu
        pygame.draw.rect(surface, (30, 30, 60), self.overlay_rect, border_radius=20)
        pygame.draw.rect(surface, WHITE, self.overlay_rect, 4, 20)
        
        # Vẽ các button
        self.resume_button.draw(surface)
        self.home_button.draw(surface)