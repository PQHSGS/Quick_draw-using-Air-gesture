# settings.py
from pathlib import Path
import numpy as np

# Kích thước màn hình
WIDTH = 1280
HEIGHT = 720
FPS = 60
IMG_SIZE = (64, 64)  # Kích thước đầu vào của model (ví dụ)
MASK_THRESHOLD = 0.5   # Ngưỡng để tạo ảnh đen trắng

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
COLUMBIABLUE = (201, 240, 255)

# Cấu hình game
TOOL_COLOR = COLUMBIABLUE
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
FONT_PATH_REGULAR = ASSETS_PATH / "arial.ttf" # Giữ lại cho text nhỏ
FONT_PATH_BOLD = ASSETS_PATH / "FC-Lilita-One-Regular.otf" # Font chính cho tiêu đề/UI
CLASSES_VN = np.array([
    'Quả táo', 'Quả chuối', 'Bánh trung thu', 'Con tàu', 'Bánh cá', 'Mặt nạ',
    'Bông hoa', 'Đèn lồng', 'Con lân', 'Ông trăng', 'Quả lê', 'Quả dứa', 'Thỏ ngọc',
    'Đèn ông sao', 'Quả dâu tây', 'Cây thần', 'Quả dưa hấu'
])

# Cấu hình UI
UI_FONT_SIZE = 32
UI_FONT_COLOR = WHITE
TARGET_FONT_SIZE = 50
TARGET_FONT_COLOR = RED
HEADER_HEIGHT = 90 # Chiều cao của thanh HUD
HEADER_COLOR = (20, 20, 20, 200) # Màu xám đen, bán trong suốt (R, G, B, Alpha)
TIMER_LOW_COLOR = (255, 50, 50) # Màu đỏ khi thời gian sắp hết
COMBO_COLOR = (50, 255, 50) # Màu xanh lá cây sáng cho combo
AVAILABLE_COLORS = {
    "Đen": (0, 0, 0),
    "Trắng": (255, 255, 255),
    "Đỏ": (255, 0, 0),
    "Xanh lá": (0, 255, 0),
    "Xanh dương": (0, 0, 255),
    "Vàng": (255, 255, 0),
    "Tím": (128, 0, 128)
}

BUTTON_WIDTH = 300
BUTTON_HEIGHT = 80
BUTTON_COLOR = (138, 255, 193) # Xanh đậm
BUTTON_HOVER_COLOR = (174, 255, 216) # Sáng hơn khi di chuột vào
BUTTON_TEXT_COLOR = BLACK
BUTTON_FONT_SIZE = 40   