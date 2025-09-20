# assets_loader.py
import pygame
from settings import ASSETS_PATH

def load_all_assets():
    """Tải tất cả hình ảnh và âm thanh cần thiết cho game."""
    assets = {
        'icons': {},
        'sounds': {}
    }
    
    # Tải icons
    try:
        correct_icon = pygame.image.load(ASSETS_PATH / "icons" / "correct.png").convert_alpha()
        incorrect_icon = pygame.image.load(ASSETS_PATH / "icons" / "incorrect.png").convert_alpha()
        assets['icons']['correct'] = pygame.transform.scale(correct_icon, (200, 200))
        assets['icons']['incorrect'] = pygame.transform.scale(incorrect_icon, (200, 200))
        assets['icons']['pause'] = pygame.image.load(ASSETS_PATH / "icons" / "pause_icon.png").convert_alpha()
        assets['icons']['play'] = pygame.image.load(ASSETS_PATH / "icons" / "play_icon.png").convert_alpha()
        assets['icons']['home'] = pygame.image.load(ASSETS_PATH / "icons" / "home_icon.png").convert_alpha()
        print("Tải icons thành công.")
    except Exception as e:
        print(f"Lỗi khi tải icons: {e}")

    # Tải âm thanh
    try:
        pygame.mixer.init() # Khởi tạo module âm thanh
        assets['sounds']['correct'] = pygame.mixer.Sound(ASSETS_PATH / "sounds" / "correct.wav")
        assets['sounds']['incorrect'] = pygame.mixer.Sound(ASSETS_PATH / "sounds" / "incorrect.wav")
        print("Tải âm thanh thành công.")
    except Exception as e:
        print(f"Lỗi khi tải âm thanh: {e}")

    return assets