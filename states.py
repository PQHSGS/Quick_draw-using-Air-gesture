import pygame
import random
import time
import numpy as np
from settings import *
from canvas import Canvas
from button import Button
from gestures import (
    is_drawing_gesture,
    is_submit_gesture,
    get_cursor_position,
    is_click_gesture
)
from ui import draw_game_hud, draw_text
from config_manager import save_settings


class State:
    def __init__(self, game):
        self.game = game
        self.cursor_pos = (0, 0)

    def update_cursor(self):
        landmarks = self.game.hand_tracker.get_landmarks()
        pos = get_cursor_position(landmarks)
        if pos:
            self.cursor_pos = pos

    def update(self):
        pass

    def draw(self, surface):
        pass

    def handle_event(self, event):
        pass

# === TRẠNG THÁI MỚI: CÀI ĐẶT ===
class SettingsState(State):
    def __init__(self, game):
        super().__init__(game)
        self.last_click_time = 0
        
        # Tạo button Quay lại
        self.back_button = Button(50, HEIGHT - 120, "Quay lại")
        
        # UI cho Thời gian
        self.time_minus_btn = pygame.Rect(WIDTH // 2, 200, 50, 50)
        self.time_plus_btn = pygame.Rect(WIDTH // 2 + 150, 200, 50, 50)

        # UI cho Màu bút
        self.brush_color_swatches = []
        for i, color in enumerate(AVAILABLE_COLORS.values()):
            rect = pygame.Rect(WIDTH // 2 - 150 + i * 60, 350, 50, 50)
            self.brush_color_swatches.append({'rect': rect, 'color': color})
            
        # UI cho Màu nền
        self.bg_color_swatches = []
        for i, color in enumerate(AVAILABLE_COLORS.values()):
            rect = pygame.Rect(WIDTH // 2 - 150 + i * 60, 500, 50, 50)
            self.bg_color_swatches.append({'rect': rect, 'color': color})

    def update(self):
        self.update_cursor()
        self.back_button.check_hover(self.cursor_pos)
        
        landmarks = self.game.hand_tracker.get_landmarks()
        is_clicking = is_click_gesture(landmarks) and (time.time() - self.last_click_time > 0.5)

        if self.back_button.is_hovered and is_clicking:
            save_settings(self.game.settings) # Lưu cài đặt
            self.game.pop_state() # Quay lại Main Menu
            return # Dừng update để tránh lỗi

        if is_clicking:
            # Xử lý nút thời gian
            if self.time_minus_btn.collidepoint(self.cursor_pos):
                self.game.settings['game_time'] = max(10, self.game.settings['game_time'] - 5)
                self.last_click_time = time.time()
            if self.time_plus_btn.collidepoint(self.cursor_pos):
                self.game.settings['game_time'] = min(300, self.game.settings['game_time'] + 5)
                self.last_click_time = time.time()
                
            # Xử lý màu bút
            for swatch in self.brush_color_swatches:
                if swatch['rect'].collidepoint(self.cursor_pos):
                    self.game.settings['brush_color'] = swatch['color']
                    self.last_click_time = time.time()
            
            # Xử lý màu nền
            for swatch in self.bg_color_swatches:
                if swatch['rect'].collidepoint(self.cursor_pos):
                    self.game.settings['background_color'] = swatch['color']
                    self.last_click_time = time.time()

    def draw(self, surface):
        self.game.draw_blurred_webcam_bg(surface)
        draw_text(surface, "Cài đặt", (WIDTH // 2, 80), FONT_PATH_BOLD, 100, WHITE)
        self.back_button.draw(surface)

        # Vẽ UI Thời gian
        draw_text(surface, "Thời gian chơi:", (WIDTH // 4, 225), FONT_PATH_REGULAR, 40, WHITE)
        pygame.draw.rect(surface, BUTTON_COLOR, self.time_minus_btn, border_radius=10)
        pygame.draw.rect(surface, BUTTON_COLOR, self.time_plus_btn, border_radius=10)
        draw_text(surface, "-", self.time_minus_btn.center, FONT_PATH_BOLD, 50, WHITE)
        draw_text(surface, "+", self.time_plus_btn.center, FONT_PATH_BOLD, 50, WHITE)
        draw_text(surface, f"{self.game.settings['game_time']}s", (WIDTH // 2 + 100, 225), FONT_PATH_BOLD, 50, WHITE)

        # Vẽ UI Màu bút
        draw_text(surface, "Màu bút vẽ:", (WIDTH // 4, 375), FONT_PATH_REGULAR, 40, WHITE)
        for swatch in self.brush_color_swatches:
            pygame.draw.rect(surface, swatch['color'], swatch['rect'], border_radius=10)
            # Vẽ viền trắng cho màu đang được chọn
            if tuple(self.game.settings['brush_color']) == swatch['color']:
                pygame.draw.rect(surface, WHITE, swatch['rect'], 4, 10)

        # Vẽ UI Màu nền
        draw_text(surface, "Màu nền:", (WIDTH // 4, 525), FONT_PATH_REGULAR, 40, WHITE)
        for swatch in self.bg_color_swatches:
            pygame.draw.rect(surface, swatch['color'], swatch['rect'], border_radius=10)
            if tuple(self.game.settings['background_color']) == swatch['color']:
                pygame.draw.rect(surface, WHITE, swatch['rect'], 4, 10)

        pygame.draw.circle(surface, YELLOW, self.cursor_pos, 10)

    def handle_event(self, event):
        # ... (Tạm thời có thể bỏ qua xử lý chuột ở đây vì đã xử lý bằng cử chỉ)
        pass


class MainMenuState(State):
    def __init__(self, game):
        super().__init__(game)
        btn_x = (WIDTH - BUTTON_WIDTH) // 2
        self.start_button = Button(btn_x, HEIGHT // 2 - 60, "Bắt đầu")
        self.settings_button = Button(btn_x, HEIGHT // 2 + 40, "Cài đặt") # Nút mới
        self.last_click_time = 0

    def update(self):
        self.update_cursor()
        self.start_button.check_hover(self.cursor_pos)
        self.settings_button.check_hover(self.cursor_pos)
        
        landmarks = self.game.hand_tracker.get_landmarks()
        is_clicking = is_click_gesture(landmarks) and (time.time() - self.last_click_time > 1)

        if is_clicking:
            if self.start_button.is_hovered:
                self.game.push_state(PlayingState(self.game))
                self.last_click_time = time.time()
            elif self.settings_button.is_hovered:
                self.game.push_state(SettingsState(self.game)) # Chuyển sang màn cài đặt
                self.last_click_time = time.time()

    def draw(self, surface):
        self.game.draw_blurred_webcam_bg(surface)
        draw_text(surface, "Drawing Game", (WIDTH // 2, HEIGHT // 4), FONT_PATH_BOLD, 120, WHITE)
        self.start_button.draw(surface)
        self.settings_button.draw(surface)
        draw_text(surface, "Dùng ngón trỏ để di chuyển, chụm 2 ngón để click", (WIDTH // 2, HEIGHT - 50), FONT_PATH_REGULAR, 24, WHITE)
        
        pygame.draw.circle(surface, YELLOW, self.cursor_pos, 10)

    def handle_event(self, event):
        # Xử lý nhấn phím 'P'
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            self.game.push_state(PlayingState(self.game))

        # MỚI: Xử lý sự kiện click chuột
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # 1 là click chuột trái
                # Chuyển đổi tọa độ chuột từ màn hình vật lý sang màn hình ảo
                phys_w, phys_h = self.game.screen.get_size()
                scale_x = WIDTH / phys_w
                scale_y = HEIGHT / phys_h
                virtual_pos = (event.pos[0] * scale_x, event.pos[1] * scale_y)

                # Kiểm tra va chạm trên màn hình ảo
                if self.start_button.rect.collidepoint(virtual_pos):
                    self.game.push_state(PlayingState(self.game))


class GameOverState(State):
    def __init__(self, game, final_score):
        super().__init__(game)
        self.final_score = final_score
        button_x = (WIDTH - BUTTON_WIDTH) // 2
        button_y = HEIGHT * 2 / 3
        self.restart_button = Button(button_x, button_y, "Chơi lại")
        self.last_click_time = 0

    def update(self):
        self.update_cursor()
        self.restart_button.check_hover(self.cursor_pos)
        
        landmarks = self.game.hand_tracker.get_landmarks()
        if self.restart_button.is_hovered and is_click_gesture(landmarks) and (time.time() - self.last_click_time > 1):
            self.game.reset_to_playing_state()
            self.last_click_time = time.time()

    def draw(self, surface):
        self.game.draw_blurred_webcam_bg(surface)
        
        draw_text(surface, "HẾT GIỜ!", (WIDTH // 2, HEIGHT // 4), FONT_PATH_BOLD, 120, RED)
        draw_text(surface, f"Điểm: {self.final_score}", (WIDTH // 2, HEIGHT // 2), FONT_PATH_BOLD, 80, WHITE)
        self.restart_button.draw(surface)
        
        pygame.draw.circle(surface, YELLOW, self.cursor_pos, 10)

    def handle_event(self, event):
        # Xử lý nhấn phím 'R'
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.game.reset_to_playing_state()

        # MỚI: Xử lý sự kiện click chuột
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                phys_w, phys_h = self.game.screen.get_size()
                scale_x = WIDTH / phys_w
                scale_y = HEIGHT / phys_h
                virtual_pos = (event.pos[0] * scale_x, event.pos[1] * scale_y)

                if self.restart_button.rect.collidepoint(virtual_pos):
                    self.game.reset_to_playing_state()


class PlayingState(State):
    def __init__(self, game):
        super().__init__(game)
        self.canvas = Canvas(CANVAS_X, CANVAS_Y, CANVAS_WIDTH, CANVAS_HEIGHT)
        self.score = 0
        self.combo = 0
        self.start_time = time.time()
        self.time_left = self.game.settings['game_time']
        self.total_time = self.game.settings['game_time'] # Lưu lại tổng thời gian
        self.brush_color = self.game.settings['brush_color']
        self.last_submit_time = 0
        self.result_icon = None
        self.result_display_end_time = 0
        
        self.score_effect_timer = 0
        self.combo_effect_timer = 0

        self.next_target()

    def next_target(self):
        self.target_id = random.randint(0, len(CLASSES_VN) - 1)
        self.target_name = CLASSES_VN[self.target_id]

    def update(self):
        self.update_cursor()
        self.time_left = self.total_time - (time.time() - self.start_time)

        if self.score_effect_timer > 0: self.score_effect_timer -= self.game.dt
        if self.combo_effect_timer > 0: self.combo_effect_timer -= self.game.dt
        
        if self.time_left <= 0:
            self.game.pop_state() 
            self.game.push_state(GameOverState(self.game, self.score))
            return

        if self.result_icon and time.time() > self.result_display_end_time:
            self.result_icon = None

        landmarks = self.game.hand_tracker.get_landmarks()
        
        if is_drawing_gesture(landmarks):
            pos = self.cursor_pos
            canvas_pos = (pos[0] - self.canvas.rect.x, pos[1] - self.canvas.rect.y)
            if not self.canvas.is_drawing: self.canvas.start_drawing(canvas_pos)
            self.canvas.draw_line(canvas_pos, BRUSH_SIZE, self.brush_color)
        else:
            if self.canvas.is_drawing: self.canvas.stop_drawing()

        if is_submit_gesture(landmarks) and (time.time() - self.last_submit_time > 2):
            surface_to_predict = self.canvas.get_surface().copy()
            if self.game.predictor.submit_for_prediction(surface_to_predict):
                self.canvas.clear()
                self.last_submit_time = time.time()
        
        result = self.game.predictor.get_latest_result()
        if result is not None:
            print("-" * 30)
            print(f"Vật thể cần vẽ: {self.target_name}")
            print("Model dự đoán:")
            for i, (name, prob) in enumerate(result):
                print(f"  {i+1}. {name} ({(prob*100):.2f}%)")

            result_ids = [np.where(CLASSES_VN == name)[0][0] for name, prob in result]

            if self.target_id in result_ids:
                print(">>> Kết quả: ĐÚNG!")
                print("-" * 30)
                self.combo += 1
                self.score += 100 * self.combo
                self.result_icon = self.game.assets['icons']['correct']
                self.game.assets['sounds']['correct'].play()
                self.score_effect_timer = 0.5
                self.combo_effect_timer = 0.5
            else:
                print(">>> Kết quả: SAI!")
                print("-" * 30)
                self.combo = 0
                self.result_icon = self.game.assets['icons']['incorrect']
                self.game.assets['sounds']['incorrect'].play()
            
            self.result_display_end_time = time.time() + 1.5
            self.game.predictor.latest_result = None 
            self.next_target()

    def draw(self, surface):
        self.canvas.draw_to_screen(surface)
        draw_game_hud(
            surface, self.score, self.combo, self.time_left, self.target_name,
            self.score_effect_timer, self.combo_effect_timer
        )
        
        if self.result_icon:
            rect = self.result_icon.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            surface.blit(self.result_icon, rect)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c: self.canvas.clear()
            if event.key == pygame.K_n: self.next_target()