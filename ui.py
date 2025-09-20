import pygame
import time
import math
from settings import *

def draw_text(surface, text, pos, font_path, font_size, color, center_aligned=True):
    try:
        font = pygame.font.Font(str(font_path), font_size)
        text_surface = font.render(text, True, color)
        if center_aligned:
            text_rect = text_surface.get_rect(center=pos)
        else:
            text_rect = text_surface.get_rect(topleft=pos)
        surface.blit(text_surface, text_rect)
    except Exception as e:
        font = pygame.font.Font(None, font_size + 10)
        text_surface = font.render(text, True, color)
        if center_aligned:
            text_rect = text_surface.get_rect(center=pos)
        else:
            text_rect = text_surface.get_rect(topleft=pos)
        surface.blit(text_surface, text_rect)

def draw_game_hud(surface, score, combo, time_left, target_text, score_effect_timer, combo_effect_timer):
    header_surface = pygame.Surface((WIDTH, HEADER_HEIGHT), pygame.SRCALPHA)
    header_surface.fill(HEADER_COLOR)
    surface.blit(header_surface, (0, 0))

    timer_pos = (80, HEADER_HEIGHT // 2)
    time_str = f"{math.ceil(time_left)}"
    timer_color = TIMER_LOW_COLOR if time_left < 10 else YELLOW
    draw_text(surface, time_str, timer_pos, FONT_PATH_BOLD, 60, timer_color)
    pygame.draw.circle(surface, timer_color, timer_pos, 45, 6)

    target_pos = (WIDTH // 2, HEADER_HEIGHT // 2)
    draw_text(surface, f"Vẽ: {target_text}", target_pos, FONT_PATH_BOLD, TARGET_FONT_SIZE, WHITE)

    score_y_pos = HEADER_HEIGHT * 0.33
    combo_y_pos = HEADER_HEIGHT * 0.66
    score_anchor = (WIDTH - 25, score_y_pos)
    combo_anchor = (WIDTH - 25, combo_y_pos)

    score_color = UI_FONT_COLOR
    score_size = UI_FONT_SIZE
    if score_effect_timer > 0:
        progress = 1 - (score_effect_timer / 0.5)
        scale = 1 + math.sin(progress * math.pi) * 0.5
        score_size = int(UI_FONT_SIZE * scale)
        score_color = YELLOW

    font_score = pygame.font.Font(str(FONT_PATH_BOLD), score_size)
    score_surf = font_score.render(f"Score: {score}", True, score_color)
    score_rect = score_surf.get_rect(right=score_anchor[0], centery=score_anchor[1])
    surface.blit(score_surf, score_rect)

    combo_color = COMBO_COLOR
    combo_size = UI_FONT_SIZE
    if combo_effect_timer > 0:
        progress = 1 - (combo_effect_timer / 0.5)
        scale = 1 + math.sin(progress * math.pi) * 0.5
        combo_size = int(UI_FONT_SIZE * scale)
    
    font_combo = pygame.font.Font(str(FONT_PATH_BOLD), combo_size)
    combo_surf = font_combo.render(f"x{combo} Combo", True, combo_color)
    combo_rect = combo_surf.get_rect(right=combo_anchor[0], centery=combo_anchor[1])
    surface.blit(combo_surf, combo_rect)