# states.py
import pygame
import random
import time
from settings import *
from canvas import Canvas
import gestures
from ui import draw_hud, draw_text

class State:
    def __init__(self, game):
        self.game = game
    def update(self): pass
    def draw(self, surface): pass
    def handle_event(self, event): pass

# === TRẠNG THÁI MỚI: MAIN MENU ===
class MainMenuState(State):
    def draw(self, surface):
        draw_text(surface, "DRAWING GAME", (WIDTH // 2, HEIGHT // 3), 100, WHITE)
        draw_text(surface, "Giơ ngón trỏ để vẽ", (WIDTH // 2, HEIGHT // 2), 50, YELLOW)
        draw_text(surface, "Xòe bàn tay để nộp bài", (WIDTH // 2, HEIGHT // 2 + 60), 50, YELLOW)
        draw_text(surface, "Nhấn 'P' để Bắt đầu", (WIDTH // 2, HEIGHT * 2 / 3 + 50), 60, GREEN)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            # Tạo một PlayingState mới và đẩy nó vào stack
            self.game.push_state(PlayingState(self.game))

# === TRẠNG THÁI MỚI: GAME OVER ===
class GameOverState(State):
    def __init__(self, game, final_score):
        super().__init__(game)
        self.final_score = final_score

    def draw(self, surface):
        draw_text(surface, "HẾT GIỜ!", (WIDTH // 2, HEIGHT // 3), 100, RED)
        draw_text(surface, f"Điểm cuối cùng: {self.final_score}", (WIDTH // 2, HEIGHT // 2), 70, WHITE)
        draw_text(surface, "Nhấn 'R' để Chơi lại", (WIDTH // 2, HEIGHT * 2 / 3 + 50), 60, GREEN)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            # Quay lại Main Menu bằng cách pop state hiện tại
            self.game.reset_to_playing_state()

# === CẬP NHẬT PLAYING STATE ===
class PlayingState(State):
    def __init__(self, game):
        super().__init__(game)
        self.canvas = Canvas(CANVAS_X, CANVAS_Y, CANVAS_WIDTH, CANVAS_HEIGHT)
        self.score = 0
        self.combo = 0
        self.start_time = time.time()
        self.time_left = TOTAL_TIME
        self.last_submit_time = 0
        
        # Biến quản lý icon kết quả
        self.result_icon = None
        self.result_display_end_time = 0

        self.next_target()

    def next_target(self):
        self.target_id = random.randint(0, len(CLASSES_VN) - 1)
        self.target_name = CLASSES_VN[self.target_id]

    def update(self):
        self.time_left = TOTAL_TIME - (time.time() - self.start_time)
        if self.time_left <= 0:
            # THAY ĐỔI: pop PlayingState hiện tại trước khi đẩy GameOverState
            # để đảm bảo stack luôn đúng
            self.game.pop_state() 
            self.game.push_state(GameOverState(self.game, self.score))
            return

        # Ẩn icon kết quả sau một khoảng thời gian
        if self.result_icon and time.time() > self.result_display_end_time:
            self.result_icon = None

        landmarks = self.game.hand_tracker.get_landmarks()
        
        # Logic vẽ
        if gestures.is_drawing_gesture(landmarks):
            pos = (landmarks[8][1], landmarks[8][2])
            canvas_pos = (pos[0] - self.canvas.rect.x, pos[1] - self.canvas.rect.y)
            if not self.canvas.is_drawing: self.canvas.start_drawing(canvas_pos)
            self.canvas.draw_line(canvas_pos, BRUSH_SIZE, TOOL_COLOR)
        else:
            if self.canvas.is_drawing: self.canvas.stop_drawing()

        # Logic nộp bài
        if gestures.is_submit_gesture(landmarks) and (time.time() - self.last_submit_time > 2):
            print("Phát hiện cử chỉ NỘP BÀI!")
            
            # THAY ĐỔI Ở ĐÂY
            # Lấy surface từ canvas và tạo một bản sao của nó
            surface_to_predict = self.canvas.get_surface().copy()
            
            # Gửi bản sao này cho predictor, không gửi bản gốc
            if self.game.predictor.submit_for_prediction(surface_to_predict):
                self.canvas.clear()
                self.last_submit_time = time.time()
        
         # Kiểm tra kết quả từ predictor
        result = self.game.predictor.get_latest_result()
        if result is not None:
            
            # THAY ĐỔI: In kết quả ra console
            print("-" * 30)
            print(f"Vật thể cần vẽ: {self.target_name}")
            print("Model dự đoán:")
            # result giờ là list các tuple (tên, xác suất)
            for i, (name, prob) in enumerate(result):
                print(f"  {i+1}. {name} ({(prob*100):.2f}%)")
            
            # Lấy ra danh sách ID từ kết quả để kiểm tra logic đúng/sai
            result_ids = [np.where(CLASSES_VN == name)[0][0] for name, prob in result]

            if self.target_id in result_ids:
                print(">>> Kết quả: ĐÚNG!")
                self.combo += 1
                self.score += 100 * self.combo
                self.result_icon = self.game.assets['icons']['correct']
                self.game.assets['sounds']['correct'].play()
            else:
                print(">>> Kết quả: SAI!")
                self.combo = 0
                self.result_icon = self.game.assets['icons']['incorrect']
                self.game.assets['sounds']['incorrect'].play()
            
            print("-" * 30)
            
            self.result_display_end_time = time.time() + 1.5
            self.game.predictor.latest_result = None 
            self.next_target()

    def draw(self, surface):
        self.canvas.draw_to_screen(surface)
        draw_hud(surface, self.score, self.combo, self.time_left, self.target_name)
        
        # Vẽ icon kết quả nếu có
        if self.result_icon:
            rect = self.result_icon.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            surface.blit(self.result_icon, rect)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c: self.canvas.clear()
            if event.key == pygame.K_n: self.next_target()