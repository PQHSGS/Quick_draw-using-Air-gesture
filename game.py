# game.py

# Mô tả: Một trò chơi "Vẽ để đoán" sử dụng cử chỉ tay điều khiển qua webcam.
#        Chương trình kết hợp OpenCV để xử lý hình ảnh, MediaPipe, PyTorch và Pygame.

'''
DANH SÁCH CỬ CHỈ:
1. (MENU) GIƠ 5 NGÓN TAY: BẮT ĐẦU GAME
2. (GAME) CHỤM 2 NGÓN TAY (TRỎ VÀ CÁI): PEN DOWN (Hạ bút)
3. (GAME) DẤU V (HI): PEN UP (Nhấc bút)
4. (GAME) 1 NGÓN TRỎ (KHI ĐÃ PEN DOWN): VẼ
5. (GAME) 5 NGÓN TAY: SUBMIT (Nộp bài)
6. (END) DẤU V (HI): CHƠI LẠI

PHÍM TẮT: Q (Thoát), C (Xóa)
'''

# ==============================================================================
# 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ==============================================================================
import time
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import pygame

from ModelArchitect import VGG_Small
from HandTrack import HandDetector

# ==============================================================================
# 2. CÁC HẰNG SỐ VÀ CẤU HÌNH
# ==============================================================================

# ---- Game Logic ----
PREDICT_COOLDOWN = 1.5
MASK_THRESHOLD = 50
BRUSH_SIZE = 25
RESULT_DISPLAY = 2.0
GAME_SIZE = 5  # Số lượng từ khóa mỗi ván
TIME_PER_ROUND = 25.0 # Thời gian (giây) cho mỗi lượt vẽ

# ---- UI Configuration ----
WIDTH, HEIGHT = 1200, 800

# Điều chỉnh vị trí khung vẽ để cân đối với UI mới
CENTER = (WIDTH // 2, HEIGHT // 2 + 50)
BOX_RANGE = 225

# Kích thước và vị trí các thành phần UI
WEBCAM_W, WEBCAM_H = 320, 240
PANEL_PADDING = 20
UI_TOP_MARGIN = 20

WEBCAM_RECT = pygame.Rect(WIDTH - WEBCAM_W - PANEL_PADDING, UI_TOP_MARGIN, WEBCAM_W, WEBCAM_H)
SCORE_PANEL_X = PANEL_PADDING

# ---- Assets Paths ----
FONT_PATH = "arial.ttf"
ICON_FOLDER = Path("icon_v2")
FONT_SIZE = 32
EMO_SIZE = 300
MODEL_PATH = Path("vgg.pt")

# ---- Pygame Colors ----
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
RED = (255, 50, 50)
GRAY = (50, 50, 50)
BG_COLOR = (20, 20, 40) # Màu nền game

CURSOR_NEUTRAL = (0, 255, 0)
CURSOR_READY = (255, 165, 0)

# ---- Game Data ----
CLASSES_VN = np.array([
    'Quả táo','Quả chuối','Bánh trung thu','Con tàu','Bánh cá','Mặt nạ',
    'Bông hoa','Đèn lồng','Con lân','Ông trăng','Quả lê','Quả dứa','Thỏ ngọc',
    'Đèn ông sao','Quả dâu tây','Cây thần','Quả dưa hấu'
])

# ==============================================================================
# 3. CÁC CẤU TRÚC DỮ LIỆU VÀ LỚP
# ==============================================================================

class Phase(Enum):
    START_MENU = auto()
    PLAYING = auto()
    GAME_OVER = auto()

@dataclass
class GameState:
    game_phase: Phase = Phase.START_MENU
    canvas: pygame.Surface = field(default_factory=lambda: pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA))
    xp: int = 0; yp: int = 0
    is_pen_down: bool = False
    last_predict_ts: float = 0.0
    last_result_ts: float = 0.0
    is_drawing: bool = False
    
    # Score and Progress
    score: int = 0; combo: int = 0; draw_count: int = 0
    
    # Timer
    round_start_time: float = 0.0

    # Target
    target_id: int = 0; target_name: str = ''
    emo_list: List[str] = field(default_factory=list)
    emo_id: np.ndarray = field(default_factory=lambda: np.array([]))
    result_icon: pygame.Surface = None

    def reset_canvas(self):
        self.canvas = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.xp = self.yp = 0
        self.is_drawing = False

    def reset_game(self):
        """Reset tất cả các chỉ số để bắt đầu một ván chơi mới."""
        print("--- BẮT ĐẦU VÁN MỚI ---")
        self.score = 0
        self.combo = 0
        self.draw_count = 0
        self.reset_canvas()
        self.result_icon = None

        # Chuẩn bị dữ liệu cho ván chơi mới
        available_indices = list(range(len(CLASSES_VN)))
        if GAME_SIZE > len(available_indices):
             self.emo_id = np.random.choice(len(CLASSES_VN), GAME_SIZE, replace=True)
        else:
             self.emo_id = np.random.choice(len(CLASSES_VN), GAME_SIZE, replace=False)

        self.emo_list = [CLASSES_VN[i] for i in self.emo_id]
        pick_new_target(self)

# ==============================================================================
# 4. CÁC HÀM TIỆN ÍCH (HELPER FUNCTIONS)
# ==============================================================================
def get_hand_gesture(detector, frame) -> str:
    landmarks = detector.lmList
    if not landmarks: return 'none'
    fingers = detector.fingersUp()
    if not fingers or len(fingers) < 5: return 'none'
    if sum(fingers) == 5: return 'submit'
    if fingers[1] and fingers[2] and not (fingers[0] or fingers[3] or fingers[4]): return 'pen_up'
    if fingers[1] and not (fingers[0] or fingers[2] or fingers[3] or fingers[4]): return 'draw'
    length, _, _ = detector.findDistance(8, 4, frame, draw=False)
    if length and length < 35: return 'pen_down'
    return 'none'

def cv2_image_to_pygame(image: np.ndarray):
    image = np.rot90(image)
    image = cv2.flip(image, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(image)

def pygame_surface_to_cv2(surface: pygame.Surface):
    img_array = pygame.surfarray.array3d(surface)
    img_bgr = cv2.cvtColor(img_array.swapaxes(0, 1), cv2.COLOR_RGB2BGR)
    return img_bgr

def load_icons(folder: Path, emo_size=300) -> Dict[int, List[pygame.Surface]]:
    """
    Tải tất cả các icon từ một thư mục và sắp xếp chúng vào một dictionary.
    Key là class_id (được lấy từ đầu tên file), value là danh sách các icon.
    YÊU CẦU: Tên file icon phải bắt đầu bằng "ID_", ví dụ: "0_apple.png".
    """
    icon_dict = {}
    if not folder.exists():
        print(f"Cảnh báo: Thư mục icon không tồn tại tại {folder}")
        return icon_dict

    # Lặp qua tất cả file trong thư mục
    for f in sorted(folder.glob("*.png")):
        if not f.is_file(): continue
        
        # Trích xuất class_id từ tên file
        try:
            class_id = int(f.name.split('_')[0])
        except (ValueError, IndexError):
            print(f"  Bỏ qua file icon có tên không hợp lệ: {f.name}")
            continue

        # Tải và xử lý ảnh (giống như trước)
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if im is None: continue
        if im.shape[2] < 4: im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        im = cv2.resize(im, (emo_size, emo_size))
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
        im = np.rot90(im); im = np.flipud(im)
        surface = pygame.image.frombuffer(im.tobytes(), im.shape[1::-1], "RGBA")
        
        # Thêm surface vào dictionary
        if class_id not in icon_dict:
            icon_dict[class_id] = []
        icon_dict[class_id].append(surface)
        
    print(f"Đã tải thành công {sum(len(v) for v in icon_dict.values())} icons cho {len(icon_dict)} lớp.")
    return icon_dict

def pick_new_target(gs: GameState):
    if gs.draw_count >= GAME_SIZE:
        gs.target_id = -1
        gs.target_name = "HOÀN THÀNH!"
        return
    
    i = gs.draw_count
    gs.target_name = gs.emo_list[i]
    gs.target_id = int(gs.emo_id[i])
    gs.round_start_time = time.time() # Reset timer cho lượt mới
    print(f"Mục tiêu mới: {gs.target_name}")


def check_draw(x, y):
    return (CENTER[0]-BOX_RANGE) <= x <= (CENTER[0]+BOX_RANGE) and \
           (CENTER[1]-BOX_RANGE) <= y <= (CENTER[1]+BOX_RANGE)

# ==============================================================================
# 5. CÁC HÀM LIÊN QUAN ĐẾN AI/PYTORCH
# ==============================================================================
# (Không thay đổi)
def torch_process_image(canvas_cv2: np.ndarray, size=(32,32)):
    gray = cv2.cvtColor(canvas_cv2, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    inp = cv2.resize(bw, size).astype(np.float32) / 255.0
    return torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)

def torch_predict(model: torch.nn.Module, image: np.ndarray, device=torch.device('cpu'), k=3):
    t = torch_process_image(image).to(device)
    with torch.no_grad():
        out = model(t)
        if out.dim() > 2: out = out.view(out.size(0), -1)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    inds = np.argsort(probs)[-k:][::-1]
    return inds

def attempt_predict(gs: GameState, model, now, icons):
    canvas_cv2 = pygame_surface_to_cv2(gs.canvas)
    y0 = max(0, CENTER[1] - BOX_RANGE); y1b = min(canvas_cv2.shape[0], CENTER[1] + BOX_RANGE)
    x0 = max(0, CENTER[0] - BOX_RANGE); x1b = min(canvas_cv2.shape[1], CENTER[0] + BOX_RANGE)
    box = canvas_cv2[y0:y1b, x0:x1b]

    class_label = np.array([-1]) if box.size == 0 or box.shape[0] < 4 or box.shape[1] < 4 else torch_predict(model, box)
    success = class_label.size > 0 and gs.target_id in class_label

    chosen_icon = None
    if success and gs.target_id in icons:
        # Lấy danh sách các icon cho lớp đã vẽ đúng
        available_icons = icons[gs.target_id]
        if available_icons:
            # Chọn ngẫu nhiên một icon từ danh sách đó
            chosen_icon = np.random.choice(available_icons)

    gs.last_predict_ts = now
    gs.result_icon = chosen_icon if success else None
    gs.last_result_ts = now

    if success:
        gs.combo += 1
        gs.score += 100 * gs.combo
    else:
        gs.combo = 0
        
    gs.draw_count += 1
    gs.reset_canvas()
    pick_new_target(gs)

def build_model(path: Path, device=torch.device('cpu')):
    if not path.exists(): raise FileNotFoundError(f"Model not found at {path}")
    model = VGG_Small(num_classes=len(CLASSES_VN))
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd.get('state_dict', sd)); model.to(device); model.eval()
    return model

# ==============================================================================
# 6. CÁC HÀM XỬ LÝ TRẠNG THÁI GAME & UI
# ==============================================================================

def draw_game_ui(screen, gs: GameState, fonts, time_left):
    """Vẽ các thành phần UI (Điểm, Timer, Từ khóa) lên màn hình."""
    font_title, font_normal = fonts

    # --- 1. Bảng Điểm (Góc trên trái) ---
    panel_y = UI_TOP_MARGIN
    
    # Điểm
    score_surf = font_normal.render(f"Điểm: {gs.score}", True, WHITE)
    screen.blit(score_surf, (SCORE_PANEL_X, panel_y))
    panel_y += font_normal.get_height() + 5

    # Combo
    combo_color = YELLOW if gs.combo > 0 else WHITE
    combo_surf = font_normal.render(f"Combo: x{gs.combo}", True, combo_color)
    screen.blit(combo_surf, (SCORE_PANEL_X, panel_y))
    panel_y += font_normal.get_height() + 5

    # Tiến trình
    progress_surf = font_normal.render(f"Tiến độ: {gs.draw_count}/{GAME_SIZE}", True, WHITE)
    screen.blit(progress_surf, (SCORE_PANEL_X, panel_y))

    # --- 2. Đồng hồ (Giữa trên) ---
    timer_color = WHITE
    if time_left < 5: timer_color = RED
    elif time_left < 10: timer_color = (255, 165, 0) # Orange

    time_text = f"{int(max(0, time_left))}"
    timer_surf = font_title.render(time_text, True, timer_color)
    timer_rect = timer_surf.get_rect(midtop=(WIDTH // 2, UI_TOP_MARGIN))
    screen.blit(timer_surf, timer_rect)

    # --- 3. Từ khóa cần vẽ ---
    target_text_surf = font_title.render(gs.target_name, True, WHITE)
    text_y = (CENTER[1] - BOX_RANGE) // 2
    text_rect = target_text_surf.get_rect(center=(WIDTH // 2, text_y))
    screen.blit(target_text_surf, text_rect)

    # --- 4. Khung vẽ ---
    draw_box_rect = pygame.Rect(CENTER[0] - BOX_RANGE, CENTER[1] - BOX_RANGE, BOX_RANGE * 2, BOX_RANGE * 2)
    pygame.draw.rect(screen, WHITE, draw_box_rect, 5, border_radius=10)
    
    # Làm tối vùng ngoài khung vẽ (Tùy chọn, để tập trung hơn)
    # overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    # overlay.fill((0, 0, 0, 100))
    # screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT) # Làm tối nền


def handle_start_menu(screen, detector, fonts, gesture):
    screen.fill(BG_COLOR)
    title_font, instruction_font = fonts
    
    title_surf = title_font.render("VẼ ĐỂ ĐOÁN", True, WHITE)
    title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100))
    screen.blit(title_surf, title_rect)
    
    inst_surf = instruction_font.render("Giơ 5 NGÓN TAY để bắt đầu", True, YELLOW)
    inst_rect = inst_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
    screen.blit(inst_surf, inst_rect)

    return Phase.PLAYING if gesture == 'submit' else Phase.START_MENU

def handle_game_over(screen, detector, gs, fonts, gesture):
    screen.fill(BG_COLOR)
    title_font, instruction_font = fonts
    
    msg = "HOÀN THÀNH!" if gs.draw_count >= GAME_SIZE else "HẾT GIỜ!"
    title_surf = title_font.render(msg, True, WHITE)
    title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 150))
    screen.blit(title_surf, title_rect)
    
    score_surf = instruction_font.render(f"Điểm của bạn: {gs.score}", True, WHITE)
    score_rect = score_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(score_surf, score_rect)
    
    inst_surf = instruction_font.render("Giơ DẤU 'V' để chơi lại", True, YELLOW)
    inst_rect = inst_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 100))
    screen.blit(inst_surf, inst_rect)

    if gesture == 'pen_up':
        gs.reset_game()
        return Phase.PLAYING
    return Phase.GAME_OVER


def handle_gameplay(screen, detector, gs: GameState, model, icons, fonts, frame, gesture):
    
    # --- 1. Kiểm tra Timer ---
    time_elapsed = time.time() - gs.round_start_time
    time_left = TIME_PER_ROUND - time_elapsed

    if time_left <= 0 and gs.draw_count < GAME_SIZE:
        print("Hết giờ!")
        # Tự động nộp bài nếu hết giờ
        if gs.is_drawing:
            attempt_predict(gs, model, time.time(), icons)
        else:
             # Nếu không vẽ gì, coi như sai và chuyển sang từ mới
             gs.draw_count += 1
             gs.combo = 0
             pick_new_target(gs)

        # Kiểm tra lại sau khi xử lý lượt cuối
        if gs.draw_count >= GAME_SIZE:
             return Phase.GAME_OVER

        # Reset timer nếu vẫn còn từ
        time_left = TIME_PER_ROUND
    
    # Kiểm tra điều kiện kết thúc game (Đã hoàn thành tất cả từ)
    if gs.draw_count >= GAME_SIZE:
        return Phase.GAME_OVER

    # --- 2. Logic Cử chỉ & Vẽ ---
    landmarks = detector.lmList
    cursor_color = CURSOR_NEUTRAL
    x1, y1 = (0, 0)
    now = time.time()

    if landmarks:
        x1, y1 = landmarks[8][1], landmarks[8][2]
        
        if gesture == 'pen_down':
            gs.is_pen_down = True
            gs.xp, gs.yp = x1, y1
        elif gesture == 'pen_up':
            gs.is_pen_down = False
            gs.xp, gs.yp = 0, 0
        elif gesture == 'draw':
            if gs.is_pen_down:
                if gs.xp == 0 and gs.yp == 0: gs.xp, gs.yp = x1, y1
                
                # Chỉ vẽ nếu ngón tay nằm trong khu vực cho phép
                if check_draw(x1, y1):
                    if check_draw(gs.xp, gs.yp): # Nếu điểm trước cũng trong khung
                         pygame.draw.line(gs.canvas, YELLOW, (gs.xp, gs.yp), (x1, y1), BRUSH_SIZE)
                    pygame.draw.circle(gs.canvas, YELLOW, (x1, y1), BRUSH_SIZE // 2)
                    gs.is_drawing = True

                gs.xp, gs.yp = x1, y1
            else:
                gs.xp, gs.yp = 0, 0
        elif gesture == 'submit':
            if gs.is_drawing and (now - gs.last_predict_ts) >= PREDICT_COOLDOWN:
                attempt_predict(gs, model, now, icons)
                gs.is_pen_down = False
        
        # Cập nhật màu con trỏ
        if gesture == 'draw' and gs.is_pen_down: cursor_color = YELLOW
        elif gs.is_pen_down: cursor_color = CURSOR_READY
        else: cursor_color = CURSOR_NEUTRAL

    # --- 3. Render (Vẽ lên màn hình) ---
    
    # 3.1. Nền
    screen.fill(BG_COLOR)

    # 3.2. Webcam (Nhỏ ở góc)
    webcam_surface = cv2_image_to_pygame(frame)
    scaled_webcam = pygame.transform.scale(webcam_surface, (WEBCAM_RECT.width, WEBCAM_RECT.height))
    screen.blit(scaled_webcam, WEBCAM_RECT)
    pygame.draw.rect(screen, WHITE, WEBCAM_RECT, 2) # Viền cho khung webcam

    # 3.3. Canvas (Vẽ đè lên trên cùng)
    screen.blit(gs.canvas, (0, 0))

    # 3.4. Giao diện UI (Điểm, Timer, Khung vẽ)
    draw_game_ui(screen, gs, fonts, time_left)

    # 3.5. Icon kết quả (Nếu có)
    if gs.result_icon is not None and (now - gs.last_result_ts) < RESULT_DISPLAY:
        icon_rect = gs.result_icon.get_rect(center=(CENTER[0], CENTER[1]))
        # Thêm nền mờ cho icon
        pygame.draw.rect(screen, BG_COLOR, icon_rect.inflate(20, 20), border_radius=10)
        screen.blit(gs.result_icon, icon_rect)

    # 3.6. Con trỏ ảo
    if landmarks:
        pygame.draw.circle(screen, WHITE, (x1, y1), BRUSH_SIZE // 2 + 3, 3)
        pygame.draw.circle(screen, cursor_color, (x1, y1), BRUSH_SIZE // 2)

    return Phase.PLAYING

# ==============================================================================
# 7. HÀM MAIN
# ==============================================================================

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Draw To Guess - Hand Tracking")
    clock = pygame.time.Clock()

    # Tải fonts
    try:
        font_title = pygame.font.Font(FONT_PATH, FONT_SIZE + 20)
        font_normal = pygame.font.Font(FONT_PATH, FONT_SIZE - 5)
        fonts = (font_title, font_normal)
    except FileNotFoundError:
        print(f"Font not found at {FONT_PATH}. Using defaults.")
        font_title = pygame.font.Font(None, FONT_SIZE + 30)
        font_normal = pygame.font.Font(None, FONT_SIZE)
        fonts = (font_title, font_normal)

    # Khởi tạo
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    detector = HandDetector(maxHands=1, detectionCon=0.7, trackCon=0.7)
    icons = load_icons(ICON_FOLDER)
    
    try:
        device = torch.device('cpu')
        model = build_model(MODEL_PATH, device)
    except FileNotFoundError as e:
        print(e)
        print("Vui lòng đảm bảo file model 'vgg.pt' tồn tại.")
        sys.exit(1)

    gs = GameState()
    current_phase = Phase.START_MENU

    # --- Vòng lặp chính ---
    running = True
    while running:
        gesture = 'none' # Reset cử chỉ mỗi frame

        # 1. Xử lý sự kiện Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: running = False
                if event.key == pygame.K_c and current_phase == Phase.PLAYING:
                    gs.reset_canvas()

        # 2. Xử lý Webcam & Hand Tracking
        ret, frame = cap.read()
        if not ret:
            print("Lỗi webcam!"); break
        frame = cv2.flip(frame, 1)
        
        detector.findHands(frame, draw=False)
        detector.findPosition(frame)
        gesture = get_hand_gesture(detector, frame)

        # 3. Game State Machine
        if current_phase == Phase.START_MENU:
            current_phase = handle_start_menu(screen, detector, fonts, gesture)
        
        elif current_phase == Phase.PLAYING:
            if gs.game_phase != Phase.PLAYING:
                gs.reset_game()
            gs.game_phase = Phase.PLAYING
            current_phase = handle_gameplay(screen, detector, gs, model, icons, fonts, frame, gesture)

        elif current_phase == Phase.GAME_OVER:
            gs.game_phase = Phase.GAME_OVER
            current_phase = handle_game_over(screen, detector, gs, fonts, gesture)

        # 4. Cập nhật màn hình
        pygame.display.flip()
        clock.tick(FPS)

    # --- Dọn dẹp ---
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()