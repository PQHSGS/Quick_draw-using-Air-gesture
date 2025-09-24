import pygame
import sys
import cv2
import numpy as np
from settings import *
from pathlib import Path
from webcam_stream import WebcamStream
from hand_tracker import HandTracker
from predictor import Predictor
from states import MainMenuState
from assets_loader import load_all_assets, load_animated_icons
from config_manager import load_settings, save_settings

def load_animated_icons(path, class_names_vn):
    """
    Tải tất cả các icon từ cấu trúc thư mục mới (icon_v2).
    Trả về một dictionary với key là class_id và value là một danh sách các icon (pygame.Surface).
    """
    animated_icons = {}
    icon_folder = Path(path)
    if not icon_folder.exists():
        print(f"Cảnh báo: Thư mục icon động không tồn tại tại {icon_folder}")
        return animated_icons

    # Tạo một bản đồ từ tên tiếng Việt sang tên thư mục (tiếng Anh, không dấu)
    # Ví dụ: 'Quả táo' -> 'apple'. Bạn cần tùy chỉnh cho khớp với tên thư mục của bạn.
    name_map = {
        'Quả táo': 'apple', 'Quả chuối': 'banana', 'Bánh trung thu': 'cake',
        'Con tàu': 'cruise_ship', 'Mặt nạ': 'face', 'Bánh cá': 'fish',
        'Bông hoa': 'flower', 'Đèn lồng': 'lantern', 'Con lân': 'lion',
        'Ông trăng': 'moon', 'Quả lê': 'pear', 'Quả dứa': 'pineapple',
        'Thỏ ngọc': 'rabbit', 'Đèn ông sao': 'star', 'Quả dâu tây': 'strawberry',
        'Cây thần': 'tree'
        # Thêm các ánh xạ khác nếu cần
    }

    # Tạo bản đồ ngược từ tên thư mục sang class_id
    folder_to_id = {}
    for i, class_name_vn in enumerate(class_names_vn):
        if class_name_vn in name_map:
            folder_name = name_map[class_name_vn]
            folder_to_id[folder_name] = i

    # Duyệt qua các thư mục con trong icon_v2
    for class_dir in icon_folder.iterdir():
        if not class_dir.is_dir():
            continue
        
        folder_name = class_dir.name
        if folder_name in folder_to_id:
            class_id = folder_to_id[folder_name]
            animated_icons[class_id] = []
            
            # Tải tất cả các ảnh trong thư mục con này
            for img_file in sorted(class_dir.glob("*.png")):
                image = pygame.image.load(img_file).convert_alpha()
                animated_icons[class_id].append(image)

    print(f"Đã tải thành công {sum(len(v) for v in animated_icons.values())} icon động cho {len(animated_icons)} lớp.")
    return animated_icons

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.virtual_screen = pygame.Surface((WIDTH, HEIGHT))
        pygame.display.set_caption("GDG Draw Game - Pygame Version")
        self.clock = pygame.time.Clock()
        self.running = True

        self.settings = load_settings()
        self.assets = load_all_assets()
        self.assets['animated_icons'] = load_animated_icons(ASSETS_PATH / "icon_v2", CLASSES_VN)

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