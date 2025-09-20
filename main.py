# main.py
import pygame
import sys
import cv2
from settings import *
from hand_tracker import HandTracker
from predictor import Predictor
from states import *
from assets_loader import load_all_assets
from webcam_stream import WebcamStream

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.virtual_screen = pygame.Surface((WIDTH, HEIGHT))
        pygame.display.set_caption("Draw Game - Pygame Version")
        self.clock = pygame.time.Clock()
        self.running = True

        # Tải tài nguyên
        self.assets = load_all_assets()

        # THAY ĐỔI: Khởi tạo các module theo thứ tự mới
        # 1. Tạo và chạy stream camera trước tiên
        self.webcam_stream = WebcamStream()
        self.webcam_stream.start()

        # 2. Tạo và chạy các module khác, truyền stream vào HandTracker
        self.hand_tracker = HandTracker(webcam_stream=self.webcam_stream)
        self.predictor = Predictor()

        # Khởi tạo các module chạy nền
        self.hand_tracker.start()
        self.predictor.start()

        # Quản lý State Stack
        self.states = []
        self.load_states()
        
        self.show_landmarks_filter = False
        self.show_webcam_feed = False

    def load_states(self):
        # Trạng thái đầu tiên là Main Menu
        self.states.append(MainMenuState(self))
        
    def push_state(self, state):
        self.states.append(state)

    def pop_state(self):
        if len(self.states) > 1: # Không pop state cuối cùng (main menu)
            self.states.pop()

     # MỚI: Hàm để reset game về một ván chơi mới
    def reset_to_playing_state(self):
        """
        Xóa tất cả các trạng thái hiện tại (trừ Main Menu) 
        và bắt đầu một PlayingState mới.
        """
        # Xóa hết các state cũ cho đến khi chỉ còn 1 state (MainMenuState)
        while len(self.states) > 1:
            self.states.pop()
        # Đẩy một ván chơi mới vào
        self.push_state(PlayingState(self))
        print("Game đã được reset để chơi lại.")

    def run(self):
        while self.running:
            self.dt = self.clock.tick(FPS) / 1000.0
            self.events()
            self.update()
            self.draw()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.quit()
            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m: pygame.display.toggle_fullscreen()
                if event.key == pygame.K_f: self.show_landmarks_filter = not self.show_landmarks_filter
                if event.key == pygame.K_w:
                    self.show_webcam_feed = not self.show_webcam_feed
                    print(f"Webcam background is now {'ON' if self.show_webcam_feed else 'OFF'}")
            
            # Gửi sự kiện cho trạng thái trên cùng xử lý
            self.states[-1].handle_event(event)

    def update(self):
        # Cập nhật trạng thái trên cùng
        self.states[-1].update()

    def draw(self):
        if self.show_webcam_feed:
            # THAY ĐỔI: Lấy frame từ stream thay vì từ hand_tracker
            frame = self.webcam_stream.get_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pygame = np.transpose(frame_rgb, (1, 0, 2))
                webcam_surface = pygame.surfarray.make_surface(frame_pygame)
                self.virtual_screen.blit(webcam_surface, (0, 0))
            else:
                self.virtual_screen.fill(GRAY)
        else:
            # Logic cũ: vẽ nền đen
            self.virtual_screen.fill(BLACK)
        
        # Vẽ trạng thái hiện tại (vẽ đè lên background)
        self.states[-1].draw(self.virtual_screen)
        
        if self.show_landmarks_filter:
            landmarks = self.hand_tracker.get_landmarks()
            if landmarks:
                for lm in landmarks: pygame.draw.circle(self.virtual_screen, GREEN, (lm[1], lm[2]), 5)
                pos = (landmarks[8][1], landmarks[8][2])
                pygame.draw.circle(self.virtual_screen, RED, pos, 15)

        scaled_surface = pygame.transform.scale(self.virtual_screen, self.screen.get_size())
        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

    def quit(self):
        # THAY ĐỔI: Dừng cả webcam_stream
        self.webcam_stream.stop()
        self.hand_tracker.stop()
        self.predictor.stop()
        # Không cần join, vì chúng là daemon threads
        pygame.quit()
        sys.exit()
        

if __name__ == '__main__':
    game = Game()
    game.run()