import pygame
import sys
import cv2
import numpy as np
from settings import *
from webcam_stream import WebcamStream
from hand_tracker import HandTracker
from predictor import Predictor
from states import MainMenuState
from assets_loader import load_all_assets
from config_manager import load_settings, save_settings

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.virtual_screen = pygame.Surface((WIDTH, HEIGHT))
        pygame.display.set_caption("Draw Game - Pygame Version")
        self.clock = pygame.time.Clock()
        self.running = True

        self.settings = load_settings()
        self.assets = load_all_assets()

        self.webcam_stream = WebcamStream()
        self.webcam_stream.start()

        self.hand_tracker = HandTracker(webcam_stream=self.webcam_stream)
        self.predictor = Predictor()
        self.hand_tracker.start()
        self.predictor.start()

        self.states = []
        self.load_states()
        
        self.show_landmarks_filter = False
        self.show_webcam_feed = True

    def load_states(self):
        self.states.append(MainMenuState(self))
        
    def push_state(self, state):
        self.states.append(state)

    def pop_state(self):
        if len(self.states) > 1:
            self.states.pop()

    def reset_to_playing_state(self):
        while len(self.states) > 1:
            self.states.pop()
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
            
            self.states[-1].handle_event(event)

    def update(self):
        self.states[-1].update()

    def draw_blurred_webcam_bg(self, surface):
        frame = self.webcam_stream.get_frame()
        if frame is not None:
            blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)
            frame_rgb = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)
            frame_pygame = np.transpose(frame_rgb, (1, 0, 2))
            webcam_surface = pygame.surfarray.make_surface(frame_pygame)
            surface.blit(webcam_surface, (0, 0))
        else:
            surface.fill(BLACK)
    
    def draw(self):
        if isinstance(self.states[-1], PlayingState) and self.show_webcam_feed:
            frame = self.webcam_stream.get_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pygame = np.transpose(frame_rgb, (1, 0, 2))
                webcam_surface = pygame.surfarray.make_surface(frame_pygame)
                self.virtual_screen.blit(webcam_surface, (0, 0))
            else:
                self.virtual_screen.fill(GRAY)
        else:
            self.virtual_screen.fill(self.settings['background_color'])
        
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
        save_settings(self.settings)
        self.webcam_stream.stop()
        self.hand_tracker.stop()
        self.predictor.stop()
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    from states import PlayingState # Import cục bộ để tránh circular import
    game = Game()
    game.run()