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

class MainMenuState(State):
    def __init__(self, game):
        super().__init__(game)
        button_x = (WIDTH - BUTTON_WIDTH) // 2
        button_y = HEIGHT // 2
        self.start_button = Button(button_x, button_y, "Bắt đầu")
        self.last_click_time = 0

    def update(self):
        self.update_cursor()
        self.start_button.check_hover(self.cursor_pos)
        
        landmarks = self.game.hand_tracker.get_landmarks()
        if self.start_button.is_hovered and is_click_gesture(landmarks) and (time.time() - self.last_click_time > 1):
            self.game.push_state(PlayingState(self.game))
            self.last_click_time = time.time()

    def draw(self, surface):
        self.game.draw_blurred_webcam_bg(surface)
        
        draw_text(surface, "Drawing Game", (WIDTH // 2, HEIGHT // 4), FONT_PATH_BOLD, 120, WHITE)
        self.start_button.draw(surface)
        draw_text(surface, "Dùng ngón trỏ để di chuyển, chụm 2 ngón để click", (WIDTH // 2, HEIGHT - 50), FONT_PATH_REGULAR, 24, WHITE)
        
        pygame.draw.circle(surface, YELLOW, self.cursor_pos, 10)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
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
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.game.reset_to_playing_state()

class PlayingState(State):
    def __init__(self, game):
        super().__init__(game)
        self.canvas = Canvas(CANVAS_X, CANVAS_Y, CANVAS_WIDTH, CANVAS_HEIGHT)
        self.score = 0
        self.combo = 0
        self.start_time = time.time()
        self.time_left = TOTAL_TIME
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
        self.time_left = TOTAL_TIME - (time.time() - self.start_time)

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
            self.canvas.draw_line(canvas_pos, BRUSH_SIZE, TOOL_COLOR)
        else:
            if self.canvas.is_drawing: self.canvas.stop_drawing()

        if is_submit_gesture(landmarks) and (time.time() - self.last_submit_time > 2):
            surface_to_predict = self.canvas.get_surface().copy()
            if self.game.predictor.submit_for_prediction(surface_to_predict):
                self.canvas.clear()
                self.last_submit_time = time.time()
        
        result = self.game.predictor.get_latest_result()
        if result is not None:
            
            # --- THÊM LẠI PHẦN LOG BỊ MẤT ---
            print("-" * 30)
            print(f"Vật thể cần vẽ: {self.target_name}")
            print("Model dự đoán:")
            for i, (name, prob) in enumerate(result):
                print(f"  {i+1}. {name} ({(prob*100):.2f}%)")
            # ------------------------------------

            result_ids = [np.where(CLASSES_VN == name)[0][0] for name, prob in result]

            if self.target_id in result_ids:
                # --- THÊM LẠI PHẦN LOG BỊ MẤT ---
                print(">>> Kết quả: ĐÚNG!")
                print("-" * 30)
                # ------------------------------------
                self.combo += 1
                self.score += 100 * self.combo
                self.result_icon = self.game.assets['icons']['correct']
                self.game.assets['sounds']['correct'].play()
                self.score_effect_timer = 0.5
                self.combo_effect_timer = 0.5
            else:
                # --- THÊM LẠI PHẦN LOG BỊ MẤT ---
                print(">>> Kết quả: SAI!")
                print("-" * 30)
                # ------------------------------------
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