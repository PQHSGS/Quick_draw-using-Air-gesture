# button.py
import pygame
from settings import *

class Button:
    def __init__(self, x, y, text):
        self.rect = pygame.Rect(x, y, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.text = text
        self.is_hovered = False
        self.font = pygame.font.Font(str(FONT_PATH_BOLD), BUTTON_FONT_SIZE)

    def draw(self, surface):
        # Chọn màu dựa trên trạng thái hover
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=15)
        
        # Vẽ text
        text_surf = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        # Kiểm tra xem vị trí (pos) có nằm trong button không
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered