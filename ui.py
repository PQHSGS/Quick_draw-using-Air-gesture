# ui.py
import pygame
import time # MỚI
import math # MỚI
from settings import *

def draw_text(surface, text, pos, font_size, color, center_aligned=True): # Thêm tùy chọn căn lề
    try:
        font = pygame.font.Font(str(FONT_PATH), font_size)
        text_surface = font.render(text, True, color)
        if center_aligned:
            text_rect = text_surface.get_rect(center=pos)
        else: # Căn lề trái
            text_rect = text_surface.get_rect(topleft=pos)
        surface.blit(text_surface, text_rect)
    except Exception as e:
        # Fallback font
        font = pygame.font.Font(None, font_size + 10)
        text_surface = font.render(text, True, color)
        if center_aligned:
            text_rect = text_surface.get_rect(center=pos)
        else:
            text_rect = text_surface.get_rect(topleft=pos)
        surface.blit(text_surface, text_rect)

def draw_game_hud(surface, score, combo, time_left, target_text):
    """Vẽ một thanh HUD hoàn chỉnh."""
    # 1. Vẽ thanh header bán trong suốt
    header_surface = pygame.Surface((WIDTH, HEADER_HEIGHT), pygame.SRCALPHA)
    header_surface.fill(HEADER_COLOR)
    surface.blit(header_surface, (0, 0))

    # 2. Vẽ đồng hồ đếm ngược (đã đúng)
    timer_pos = (80, HEADER_HEIGHT // 2)
    time_str = f"{math.ceil(time_left)}"
    timer_color = TIMER_LOW_COLOR if time_left < 10 else YELLOW
    draw_text(surface, time_str, timer_pos, 60, timer_color)
    pygame.draw.circle(surface, timer_color, timer_pos, 45, 6)

    # 3. Vẽ vật thể cần vẽ (ở giữa, đã đúng)
    target_pos = (WIDTH // 2, HEADER_HEIGHT // 2)
    draw_text(surface, f"Vẽ: {target_text}", target_pos, TARGET_FONT_SIZE, TARGET_FONT_COLOR)

    # 4. THAY ĐỔI: Vẽ Điểm và Combo với cách căn lề mới
    # Tính toán vị trí Y cho hai dòng text một cách linh hoạt
    y_pos_score = HEADER_HEIGHT * 0.33  # Vị trí 1/3 từ trên xuống của header
    y_pos_combo = HEADER_HEIGHT * 0.70  # Vị trí 7/10 từ trên xuống của header

    # Tọa độ điểm neo bên phải của header
    score_anchor_pos = (WIDTH - 25, y_pos_score)
    combo_anchor_pos = (WIDTH - 25, y_pos_combo)

    # Vẽ text Score, căn lề phải theo chiều ngang, và giữa theo chiều dọc của điểm neo
    font_score = pygame.font.Font(str(FONT_PATH), UI_FONT_SIZE)
    score_surf = font_score.render(f"Score: {score}", True, UI_FONT_COLOR)
    score_rect = score_surf.get_rect(right=score_anchor_pos[0], centery=score_anchor_pos[1])
    surface.blit(score_surf, score_rect)
    
    # Vẽ text Combo, tương tự
    font_combo = pygame.font.Font(str(FONT_PATH), UI_FONT_SIZE)
    combo_surf = font_combo.render(f"x{combo} Combo", True, COMBO_COLOR)
    combo_rect = combo_surf.get_rect(right=combo_anchor_pos[0], centery=combo_anchor_pos[1])
    surface.blit(combo_surf, combo_rect)

# NÂNG CẤP CÁC MÀN HÌNH MENU
def draw_main_menu(surface):
    draw_text(surface, "DRAWING GAME", (WIDTH // 2, HEIGHT // 3 - 50), 100, WHITE)
    draw_text(surface, "Giơ ngón trỏ để vẽ", (WIDTH // 2, HEIGHT // 2), 50, YELLOW)
    draw_text(surface, "Xòe bàn tay để nộp bài", (WIDTH // 2, HEIGHT // 2 + 60), 50, YELLOW)
    
    # Hiệu ứng nhấp nháy cho chữ "Bắt đầu"
    pulse = math.sin(time.time() * 5) > 0
    start_color = GREEN if pulse else WHITE
    draw_text(surface, "Nhấn 'P' để Bắt đầu", (WIDTH // 2, HEIGHT * 2 / 3 + 50), 60, start_color)

def draw_game_over(surface, final_score):
    draw_text(surface, "HẾT GIỜ!", (WIDTH // 2, HEIGHT // 3), 100, RED)
    draw_text(surface, f"Điểm cuối cùng: {final_score}", (WIDTH // 2, HEIGHT // 2), 70, WHITE)
    
    pulse = math.sin(time.time() * 5) > 0
    restart_color = GREEN if pulse else WHITE
    draw_text(surface, "Nhấn 'R' để Chơi lại", (WIDTH // 2, HEIGHT * 2 / 3 + 50), 60, restart_color)