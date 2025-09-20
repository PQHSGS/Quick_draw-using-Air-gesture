# settings.py
from pathlib import Path
import numpy as np

# Kích thước màn hình
WIDTH = 1280
HEIGHT = 720
FPS = 60

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# Cấu hình game
TOOL_COLOR = YELLOW
BRUSH_SIZE = 12
TOTAL_TIME = 60 # Tổng thời gian một màn chơi (giây)

# Cấu hình Canvas
BOX_RANGE = 225
CANVAS_WIDTH = BOX_RANGE * 2
CANVAS_HEIGHT = BOX_RANGE * 2
CANVAS_X = (WIDTH - CANVAS_WIDTH) // 2
CANVAS_Y = (HEIGHT - CANVAS_HEIGHT) // 2

# Cấu hình Model & Assets
ASSETS_PATH = Path("assets") # Thư mục tài nguyên chung
MODEL_PATH = ASSETS_PATH / "vgg.pt"
FONT_PATH = ASSETS_PATH / "arial.ttf"
CLASSES_VN = np.array([
    'Quả táo', 'Quả chuối', 'Bánh trung thu', 'Con tàu', 'Bánh cá', 'Mặt nạ',
    'Bông hoa', 'Đèn lồng', 'Con lân', 'Ông trăng', 'Quả lê', 'Quả dứa', 'Thỏ ngọc',
    'Đèn ông sao', 'Quả dâu tây', 'Cây thần', 'Quả dưa hấu'
])

# Cấu hình UI
UI_FONT_SIZE = 36
UI_FONT_COLOR = WHITE
TARGET_FONT_SIZE = 50
TARGET_FONT_COLOR = RED