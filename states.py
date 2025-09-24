import pygame
import random
import time
import numpy as np
from settings import *
from canvas import Canvas
from button import Button
# Import cả AnimatedIcon và StaticIcon từ cùng một file
from animated_icon import AnimatedIcon, StaticIcon 
from gestures import (
    is_drawing_gesture,
    is_submit_gesture,
    get_cursor_position,
    is_click_gesture
)
from ui import draw_game_hud, draw_text
from config_manager import save_settings
from pause_menu import PauseMenu


class State:
    def __init__(self, game):
        self.game = game
        self.cursor_pos = (0, 0)

    def update_cursor(self):
        landmarks = self.game.hand_tracker.get_landmarks()
        pos = get_cursor_position(landmarks)
        if pos:
            self.cursor_pos = pos

    def get_virtual_mouse_pos(self, physical_pos):
        phys_w, phys_h = self.game.screen.get_size()
        scale_x = WIDTH / phys_w
        scale_y = HEIGHT / phys_h
        return (physical_pos[0] * scale_x, physical_pos[1] * scale_y)

    def update(self):
        pass

    def draw(self, surface):
        pass

    def handle_event(self, event):
        pass

class SettingsState(State):
    def __init__(self, game):
        super().__init__(game)
        self.last_click_time = 0
        self.back_button = Button(50, HEIGHT - 120, BUTTON_WIDTH, BUTTON_HEIGHT, text="Quay lại")
        self.time_minus_btn = pygame.Rect(WIDTH // 2, 200, 50, 50)
        self.time_plus_btn = pygame.Rect(WIDTH // 2 + 150, 200, 50, 50)
        self.brush_color_swatches = []
        for i, color in enumerate(AVAILABLE_COLORS.values()):
            rect = pygame.Rect(WIDTH // 2 - 150 + i * 60, 350, 50, 50)
            self.brush_color_swatches.append({'rect': rect, 'color': color})
        self.bg_color_swatches = []
        for i, color in enumerate(AVAILABLE_COLORS.values()):
            rect = pygame.Rect(WIDTH // 2 - 150 + i * 60, 500, 50, 50)
            self.bg_color_swatches.append({'rect': rect, 'color': color})

    def update(self):
        self.update_cursor()
        self.back_button.check_hover(self.cursor_pos)
        landmarks = self.game.hand_tracker.get_landmarks()
        is_clicking = is_click_gesture(landmarks)
        if self.back_button.update_hold(is_clicking):
            save_settings(self.game.settings)
            self.game.pop_state()
            return
        is_instant_click = is_clicking and (time.time() - self.last_click_time > 0.5)
        if is_instant_click:
            self.check_interaction(self.cursor_pos)

    def check_interaction(self, pos):
        if self.time_minus_btn.collidepoint(pos):
            self.game.settings['game_time'] = max(10, self.game.settings['game_time'] - 5)
            self.last_click_time = time.time()
        if self.time_plus_btn.collidepoint(pos):
            self.game.settings['game_time'] = min(300, self.game.settings['game_time'] + 5)
            self.last_click_time = time.time()
        for swatch in self.brush_color_swatches:
            if swatch['rect'].collidepoint(pos):
                self.game.settings['brush_color'] = swatch['color']
                self.last_click_time = time.time()
        for swatch in self.bg_color_swatches:
            if swatch['rect'].collidepoint(pos):
                self.game.settings['background_color'] = swatch['color']
                self.last_click_time = time.time()

    def draw(self, surface):
        self.game.draw_blurred_webcam_bg(surface)
        draw_text(surface, "Cài đặt", (WIDTH // 2, 80), FONT_PATH_BOLD, 100, WHITE)
        self.back_button.draw(surface)
        draw_text(surface, "Thời gian chơi:", (WIDTH // 4, 225), FONT_PATH_REGULAR, 40, WHITE)
        pygame.draw.rect(surface, BUTTON_COLOR, self.time_minus_btn, border_radius=10)
        pygame.draw.rect(surface, BUTTON_COLOR, self.time_plus_btn, border_radius=10)
        draw_text(surface, "-", self.time_minus_btn.center, FONT_PATH_BOLD, 50, WHITE)
        draw_text(surface, "+", self.time_plus_btn.center, FONT_PATH_BOLD, 50, WHITE)
        draw_text(surface, f"{self.game.settings['game_time']}s", (WIDTH // 2 + 100, 225), FONT_PATH_BOLD, 50, WHITE)
        draw_text(surface, "Màu bút vẽ:", (WIDTH // 4, 375), FONT_PATH_REGULAR, 40, WHITE)
        for swatch in self.brush_color_swatches:
            pygame.draw.rect(surface, swatch['color'], swatch['rect'], border_radius=10)
            if tuple(self.game.settings['brush_color']) == swatch['color']:
                pygame.draw.rect(surface, WHITE, swatch['rect'], 4, 10)
        draw_text(surface, "Màu nền:", (WIDTH // 4, 525), FONT_PATH_REGULAR, 40, WHITE)
        for swatch in self.bg_color_swatches:
            pygame.draw.rect(surface, swatch['color'], swatch['rect'], border_radius=10)
            if tuple(self.game.settings['background_color']) == swatch['color']:
                pygame.draw.rect(surface, WHITE, swatch['rect'], 4, 10)
        pygame.draw.circle(surface, YELLOW, self.cursor_pos, 10)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            virtual_pos = self.get_virtual_mouse_pos(event.pos)
            if self.back_button.rect.collidepoint(virtual_pos):
                save_settings(self.game.settings)
                self.game.pop_state()
                return
            self.check_interaction(virtual_pos)


class MainMenuState(State):
    def __init__(self, game):
        super().__init__(game)
        btn_x = (WIDTH - BUTTON_WIDTH) // 2
        self.start_button = Button(btn_x, HEIGHT // 2 - 60, BUTTON_WIDTH, BUTTON_HEIGHT, text="Bắt đầu")
        self.settings_button = Button(btn_x, HEIGHT // 2 + 40, BUTTON_WIDTH, BUTTON_HEIGHT, text="Cài đặt")

    def update(self):
        self.update_cursor()
        self.start_button.check_hover(self.cursor_pos)
        self.settings_button.check_hover(self.cursor_pos)
        landmarks = self.game.hand_tracker.get_landmarks()
        is_clicking = is_click_gesture(landmarks)
        if self.start_button.update_hold(is_clicking):
            self.game.push_state(InstructionsState(self.game))
        if self.settings_button.update_hold(is_clicking):
            self.game.push_state(SettingsState(self.game))

    def draw(self, surface):
        self.game.draw_blurred_webcam_bg(surface)
        draw_text(surface, "GDG Drawing Game", (WIDTH // 2, HEIGHT // 4), FONT_PATH_BOLD, 120, WHITE)
        self.start_button.draw(surface)
        self.settings_button.draw(surface)
        draw_text(surface, "Dùng ngón trỏ để di chuyển, chụm 2 ngón để GIỮ CHỌN", (WIDTH // 2, HEIGHT - 50), FONT_PATH_REGULAR, 24, WHITE)
        pygame.draw.circle(surface, YELLOW, self.cursor_pos, 10)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            self.game.push_state(InstructionsState(self.game))
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            virtual_pos = self.get_virtual_mouse_pos(event.pos)
            if self.start_button.rect.collidepoint(virtual_pos):
                self.game.push_state(InstructionsState(self.game))
            elif self.settings_button.rect.collidepoint(virtual_pos):
                 self.game.push_state(SettingsState(self.game))


class InstructionsState(State):
    def __init__(self, game):
        super().__init__(game)
        btn_x = (WIDTH - BUTTON_WIDTH) // 2
        self.play_button = Button(btn_x, HEIGHT - 150, BUTTON_WIDTH, BUTTON_HEIGHT, text="Chơi!")

    def update(self):
        self.update_cursor()
        self.play_button.check_hover(self.cursor_pos)
        landmarks = self.game.hand_tracker.get_landmarks()
        is_clicking = is_click_gesture(landmarks)
        if self.play_button.update_hold(is_clicking):
            self.game.pop_state()
            self.game.push_state(PlayingState(self.game))

    def draw(self, surface):
        self.game.draw_blurred_webcam_bg(surface)
        draw_text(surface, "Hướng Dẫn Chơi", (WIDTH // 2, 100), FONT_PATH_BOLD, 90, WHITE)
        panel_padding = 40
        panel_width = 800
        panel_height = 400
        panel_x = (WIDTH - panel_width) / 2
        panel_y = 200
        panel_border_radius = 20
        panel_bg_color = (0, 0, 0, 150)
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, panel_bg_color, panel_surface.get_rect(), border_radius=panel_border_radius)
        surface.blit(panel_surface, (panel_x, panel_y))
        instructions = [
            "- Giơ 1 ngón trỏ để VẼ theo vật thể yêu cầu.",
            "- Xòe cả 5 ngón tay để NỘP BÀI.",
            "- Chụm ngón trỏ và ngón giữa để GIỮ CHỌN nút.",
            "- Cố gắng vẽ đúng liên tiếp để đạt COMBO cao!"
        ]
        start_y_text = panel_y + panel_padding + 30
        for i, text in enumerate(instructions):
            draw_text(surface, text, (WIDTH // 2, start_y_text + i * 70), FONT_PATH_REGULAR, 30, WHITE)
        self.play_button.draw(surface)
        pygame.draw.circle(surface, YELLOW, self.cursor_pos, 10)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and (event.key == pygame.K_p or event.key == pygame.K_RETURN):
            self.game.pop_state()
            self.game.push_state(PlayingState(self.game))
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            virtual_pos = self.get_virtual_mouse_pos(event.pos)
            if self.play_button.rect.collidepoint(virtual_pos):
                self.game.pop_state()
                self.game.push_state(PlayingState(self.game))


class GameOverState(State):
    def __init__(self, game, final_score):
        super().__init__(game)
        self.final_score = final_score
        button_x = (WIDTH - BUTTON_WIDTH) // 2
        button_y = HEIGHT * 2 / 3
        self.restart_button = Button(button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT, text="Chơi lại")

    def update(self):
        self.update_cursor()
        self.restart_button.check_hover(self.cursor_pos)
        landmarks = self.game.hand_tracker.get_landmarks()
        is_clicking = is_click_gesture(landmarks)
        if self.restart_button.update_hold(is_clicking):
            self.game.reset_to_playing_state()

    def draw(self, surface):
        self.game.draw_blurred_webcam_bg(surface)
        draw_text(surface, "HẾT GIỜ!", (WIDTH // 2, HEIGHT // 4), FONT_PATH_BOLD, 120, RED)
        draw_text(surface, f"Điểm: {self.final_score}", (WIDTH // 2, HEIGHT // 2), FONT_PATH_BOLD, 80, WHITE)
        self.restart_button.draw(surface)
        pygame.draw.circle(surface, YELLOW, self.cursor_pos, 10)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.game.reset_to_playing_state()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            virtual_pos = self.get_virtual_mouse_pos(event.pos)
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
        self.total_time = self.game.settings['game_time']
        self.brush_color = self.game.settings['brush_color']
        self.animated_result_icon = None
        self.score_effect_timer = 0
        self.combo_effect_timer = 0
        self.last_click_time = 0 
        self.last_submit_time = 0
        self.is_paused = False 
        self.pause_menu = PauseMenu(game)
        
        # Thêm các biến quản lý trạng thái Gợi ý
        self.show_hint = False
        self.current_hint_image = None

        icon_size = 50
        padding = 250
        pause_x = WIDTH // 2 - padding - icon_size
        pause_y = (HEADER_HEIGHT - icon_size) // 2
        self.pause_button = Button(pause_x, pause_y, icon_size, icon_size, icon=self.game.assets['icons']['pause'])
        home_x = WIDTH // 2 + padding
        home_y = (HEADER_HEIGHT - icon_size) // 2
        self.home_button_hud = Button(home_x, home_y, icon_size, icon_size, icon=self.game.assets['icons']['home'])
        self.next_target()

    def next_target(self):
        self.target_id = random.randint(0, len(CLASSES_VN) - 1)
        self.target_name = CLASSES_VN[self.target_id]
        
        # Cập nhật ảnh gợi ý mỗi khi có mục tiêu mới
        if self.target_id in self.game.assets['animated_icons']:
            icon_list = self.game.assets['animated_icons'][self.target_id]
            if icon_list:
                chosen_hint_icon = random.choice(icon_list)
                self.current_hint_image = pygame.transform.smoothscale(chosen_hint_icon, (150, 150))
        else:
            self.current_hint_image = None
        
        # Luôn tắt gợi ý khi sang câu mới
        self.show_hint = False

    def update(self):
        if self.animated_result_icon:
            self.animated_result_icon.update()
            if self.animated_result_icon.is_finished:
                self.animated_result_icon = None
        if self.is_paused:
            self.update_cursor()
            landmarks = self.game.hand_tracker.get_landmarks()
            is_clicking = is_click_gesture(landmarks) and (time.time() - self.last_click_time > 0.5)
            action = self.pause_menu.handle_input(self.cursor_pos, is_clicking)
            if action:
                self.handle_pause_menu_action(action)
            return
        self.update_cursor()
        self.pause_button.check_hover(self.cursor_pos)
        self.home_button_hud.check_hover(self.cursor_pos)
        landmarks = self.game.hand_tracker.get_landmarks()
        is_clicking_hud = is_click_gesture(landmarks) and (time.time() - self.last_click_time > 0.5)
        if is_clicking_hud:
            if self.pause_button.is_hovered:
                self.toggle_pause()
            elif self.home_button_hud.is_hovered:
                self.game.pop_state()
                return
        self.time_left = self.total_time - (time.time() - self.start_time)
        if self.score_effect_timer > 0: self.score_effect_timer -= self.game.dt
        if self.combo_effect_timer > 0: self.combo_effect_timer -= self.game.dt
        if self.time_left <= 0:
            self.game.pop_state() 
            self.game.push_state(GameOverState(self.game, self.score))
            return
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
                
                if self.target_id in self.game.assets['animated_icons']:
                    icon_list = self.game.assets['animated_icons'][self.target_id]
                    if icon_list:
                        self.animated_result_icon = AnimatedIcon(icon_list)
                    else:
                        correct_icon = self.game.assets['icons'].get('correct')
                        if correct_icon:
                            self.animated_result_icon = StaticIcon(correct_icon)
                else:
                    correct_icon = self.game.assets['icons'].get('correct')
                    if correct_icon:
                        self.animated_result_icon = StaticIcon(correct_icon)
            
                self.game.assets['sounds']['correct'].play()
                self.score_effect_timer = 0.5
                self.combo_effect_timer = 0.5
            else:
                print(">>> Kết quả: SAI!")
                print("-" * 30)
                self.combo = 0
                
                incorrect_icon = self.game.assets['icons'].get('incorrect')
                if incorrect_icon:
                    self.animated_result_icon = StaticIcon(incorrect_icon)
                
                self.game.assets['sounds']['incorrect'].play()
            
            self.game.predictor.latest_result = None 
            self.next_target()
    
    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_time = time.time()
        else:
            if hasattr(self, 'pause_time'):
                self.start_time += (time.time() - self.pause_time)
        self.pause_button.icon = self.game.assets['icons']['play'] if self.is_paused else self.game.assets['icons']['pause']
        self.last_click_time = time.time()

    def handle_pause_menu_action(self, action):
        self.last_click_time = time.time()
        if action == "resume":
            self.toggle_pause()
        elif action == "go_home":
            self.game.pop_state()

    def draw(self, surface):
        self.canvas.draw_to_screen(surface)
        draw_game_hud(
            surface, self.score, self.combo, self.time_left, self.target_name,
            self.score_effect_timer, self.combo_effect_timer,
            self.pause_button, self.home_button_hud
        )
        
        if self.animated_result_icon:
            self.animated_result_icon.draw(surface)
        
        # Vẽ ảnh gợi ý nếu được kích hoạt
        if self.show_hint and self.current_hint_image:
            hint_x = self.canvas.rect.right + 20 + self.current_hint_image.get_width() / 2
            hint_y = self.canvas.rect.centery
            hint_rect = self.current_hint_image.get_rect(center=(hint_x, hint_y))
            bg_rect = hint_rect.inflate(20, 20)
            pygame.draw.rect(surface, (0, 0, 0, 120), bg_rect, border_radius=15)
            surface.blit(self.current_hint_image, hint_rect)
        
        if self.is_paused:
            self.pause_menu.draw(surface)
        
        if not self.is_paused:
             pygame.draw.circle(surface, YELLOW, self.cursor_pos, 10)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
            self.show_hint = not self.show_hint

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                self.toggle_pause()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            virtual_pos = self.get_virtual_mouse_pos(event.pos)
            if self.is_paused:
                action = self.pause_menu.handle_input(virtual_pos, True)
                if action:
                    self.handle_pause_menu_action(action)
            else:
                if self.pause_button.rect.collidepoint(virtual_pos):
                    self.toggle_pause()
                elif self.home_button_hud.rect.collidepoint(virtual_pos):
                    self.game.pop_state()

        if self.is_paused:
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c: self.canvas.clear()
            if event.key == pygame.K_n: self.next_target()