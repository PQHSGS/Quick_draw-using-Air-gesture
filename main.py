# main.py
import pygame
import sys
from settings import *
from hand_tracker import HandTracker
from predictor import Predictor
from states import MainMenuState # Chỉ cần import state đầu tiên
from assets_loader import load_all_assets

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

        # Khởi tạo các module chạy nền
        self.hand_tracker = HandTracker()
        self.predictor = Predictor()
        self.hand_tracker.start()
        self.predictor.start()

        # Quản lý State Stack
        self.states = []
        self.load_states()
        
        self.show_landmarks_filter = False

    def load_states(self):
        # Trạng thái đầu tiên là Main Menu
        self.states.append(MainMenuState(self))
        
    def push_state(self, state):
        self.states.append(state)

    def pop_state(self):
        if len(self.states) > 1: # Không pop state cuối cùng (main menu)
            self.states.pop()

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
            
            # Gửi sự kiện cho trạng thái trên cùng xử lý
            self.states[-1].handle_event(event)

    def update(self):
        # Cập nhật trạng thái trên cùng
        self.states[-1].update()

    def draw(self):
        self.virtual_screen.fill(BLACK)
        
        # Vẽ trạng thái trên cùng
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
        self.hand_tracker.stop()
        self.predictor.stop()
        self.hand_tracker.join(timeout=1)
        self.predictor.join(timeout=1)
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    game = Game()
    game.run()