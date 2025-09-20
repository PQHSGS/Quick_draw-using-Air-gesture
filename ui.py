# ui.py
import pygame
from settings import *

def draw_text(surface, text, pos, font_size, color):
    try:
        font = pygame.font.Font(str(FONT_PATH), font_size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)
        surface.blit(text_surface, text_rect)
    except Exception as e:
        print(f"Lỗi font: {e}. Đang dùng font mặc định.")
        font = pygame.font.Font(None, font_size + 10) # Dùng font mặc định của Pygame
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)
        surface.blit(text_surface, text_rect)

def draw_hud(surface, score, combo, time_left, target_text):
    # Vẽ điểm
    draw_text(surface, f"Score: {score}", (WIDTH - 150, 50), UI_FONT_SIZE, UI_FONT_COLOR)
    # Vẽ combo
    draw_text(surface, f"x{combo} Combo", (WIDTH - 150, 100), UI_FONT_SIZE, GREEN)
    # Vẽ thời gian
    time_str = f"{int(time_left)}"
    draw_text(surface, time_str, (80, 60), 60, YELLOW)
    pygame.draw.circle(surface, YELLOW, (80, 60), 50, 5)
    # Vẽ vật thể cần vẽ
    draw_text(surface, f"Vẽ: {target_text}", (WIDTH // 2, 50), TARGET_FONT_SIZE, TARGET_FONT_COLOR)