import pygame
from settings import *

class Button:
    # THAY ĐỔI: Thêm các tham số mới và cho phép icon
    def __init__(self, x, y, width, height, text=None, icon=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.icon = icon
        self.is_hovered = False
        if self.text:
            self.font = pygame.font.Font(str(FONT_PATH_BOLD), BUTTON_FONT_SIZE)

    def draw(self, surface):
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=15)
        
        # Vẽ text hoặc icon
        if self.text:
            text_surf = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)
        elif self.icon:
            # Scale icon cho vừa với button (giữ lại một chút padding)
            icon_size = int(self.rect.height * 0.7)
            scaled_icon = pygame.transform.smoothscale(self.icon, (icon_size, icon_size))
            icon_rect = scaled_icon.get_rect(center=self.rect.center)
            surface.blit(scaled_icon, icon_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered