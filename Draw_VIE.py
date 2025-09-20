# Draw_VIE.py

# Import các thư viện cần thiết
import os
import time
from dataclasses import dataclass, field
from pathlib import Path # Thư viện để làm việc với đường dẫn file một cách hiệu quả
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont # Thư viện Pillow để xử lý ảnh và vẽ text Unicode
import mediapipe as mp

from ModelArchitect import VGG_Small # Import kiến trúc model VGG từ file cục bộ

# --------------------------
# Cấu hình toàn cục (Configuration)
# --------------------------
FRAME_SKIP = 2 # Bỏ qua frame để giảm tải xử lý, chỉ xử lý 1 trong 2 frame
WIDTH = 1200 # Chiều rộng cửa sổ game
HEIGHT = 800 # Chiều cao cửa sổ game
OFFSET = 20
LIMIT = 600
CENTER = (WIDTH // 2, HEIGHT // 2) # Tọa độ trung tâm màn hình
BOX_RANGE = 225 # Kích thước của khung vẽ (nửa chiều rộng/cao)
IMG_SIZE = 120
EMO_SIZE = 300 # Kích thước icon kết quả
NUM_PER_CLASS = 3 # Số lượng icon cho mỗi lớp (class)
FONT_PATH = "arial.ttf" # Đường dẫn đến file font chữ (hỗ trợ Unicode)
FONT_SIZE = 45 # Cỡ chữ
ICON_FOLDER = Path("icon_v2") # Thư mục chứa các icon của vật thể
MODEL_PATH = Path("vgg.pt") # Đường dẫn đến file model đã huấn luyện
LOGO_PATH = Path("vme.jpg") # Đường dẫn đến file logo

# Danh sách tên các lớp (vật thể) bằng tiếng Việt
CLASSES_VN = np.array([
    'Quả táo', 'Quả chuối', 'Bánh trung thu', 'Con tàu', 'Bánh cá', 'Mặt nạ',
    'Bông hoa', 'Đèn lồng', 'Con lân', 'Ông trăng', 'Quả lê', 'Quả dứa', 'Thỏ ngọc',
    'Đèn ông sao', 'Quả dâu tây', 'Cây thần', 'Quả dưa hấu'
])

# --------------------------
# Các hàm hỗ trợ (Helpers)
# --------------------------

# Hàm tải logo từ đường dẫn và thay đổi kích thước
def load_logo(path: Path, size: Tuple[int, int] = (70, 70)) -> Optional[np.ndarray]:
    if not path.exists(): # Kiểm tra file có tồn tại không
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) # Đọc ảnh, giữ nguyên kênh alpha (nếu có)
    if img is None:
        return None
    return cv2.resize(img, size) # Thay đổi kích thước và trả về

# Hàm tải tất cả các icon từ thư mục
def load_icons(folder: Path, emo_size: int = EMO_SIZE) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    if not folder.exists(): # Kiểm tra thư mục có tồn tại không
        return []
    icons = []
    # Lặp qua các thư mục con (mỗi thư mục là một lớp)
    for cls in sorted(folder.iterdir()):
        if not cls.is_dir():
            continue
        # Lặp qua các file ảnh trong thư mục con
        for f in sorted(cls.iterdir()):
            im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            im = cv2.resize(im, (emo_size, emo_size)) # Thay đổi kích thước icon
            # Xử lý ảnh có kênh trong suốt (alpha)
            if im.ndim == 3 and im.shape[2] == 4:
                rgb = im[:, :, :3] # Tách kênh màu RGB
                alpha = im[:, :, 3] # Tách kênh alpha
                icons.append((rgb, alpha))
            else:
                # Xử lý ảnh không có kênh alpha
                if im.ndim == 2: # Nếu là ảnh xám
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                icons.append((im, None))
    return icons

# Hàm dán một ảnh (overlay) lên ảnh nền (base) một cách an toàn, tránh lỗi tràn viền
def paste_image_safe(base: np.ndarray, overlay, topleft: Tuple[int, int]):
    if overlay is None:
        return
    # Tách kênh màu và kênh alpha nếu có
    if isinstance(overlay, tuple):
        rgb, alpha = overlay
        if alpha is not None:
            overlay_arr = np.dstack([rgb, alpha])
        else:
            overlay_arr = rgb
    else:
        overlay_arr = overlay
    
    if overlay_arr.ndim < 3:
        return
        
    bh, bw = base.shape[:2] # Kích thước ảnh nền
    th, tw = overlay_arr.shape[:2] # Kích thước ảnh dán
    x, y = topleft # Tọa độ góc trên bên trái
    
    # Tính toán vùng giao nhau (intersection) để tránh tràn ra ngoài
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + tw)
    y2 = min(bh, y + th)
    
    if x1 >= x2 or y1 >= y2:
        return
        
    # Tính toán vùng cần cắt từ ảnh dán
    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)
    
    # Lấy vùng quan tâm (Region of Interest) từ ảnh nền
    roi = base[y1:y2, x1:x2]
    # Cắt phần tương ứng từ ảnh dán
    patch = overlay_arr[oy1:oy2, ox1:ox2]
    
    # Trộn ảnh nếu có kênh alpha
    if patch.shape[2] == 4:
        alpha = patch[:, :, 3].astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        rgb_patch = patch[:, :, :3].astype(np.float32)
        # Công thức trộn alpha blending
        roi[:] = (alpha * rgb_patch + (1 - alpha) * roi.astype(np.float32)).astype(np.uint8)
    else:
        # Trộn ảnh theo trọng số nếu không có kênh alpha
        base[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.3, patch[:, :, :3], 0.7, 0)


# Hàm vẽ text Unicode (tiếng Việt) lên ảnh, có fallback về cv2.putText nếu lỗi
def put_text_unicode(img: np.ndarray, text: str, position: Tuple[int, int], font_path: str = FONT_PATH,
                     font_size: int = FONT_SIZE, color=(255, 255, 255)) -> np.ndarray:
    try:
        # Chuyển ảnh từ OpenCV (BGR) sang Pillow (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # Tải font chữ
        font = ImageFont.truetype(font_path, font_size)
        # Vẽ text lên ảnh
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        # Chuyển ảnh trở lại định dạng OpenCV
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # Nếu có lỗi (ví dụ không tìm thấy font), dùng hàm cơ bản của OpenCV (không hiển thị tiếng Việt)
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color, 2)
        return img

# -- Tiền xử lý ảnh và dự đoán bằng PyTorch --

# Hàm tiền xử lý ảnh vẽ tay để đưa vào model
def torch_process_image(canvas: np.ndarray, size: Tuple[int, int] = (32, 32)) -> torch.Tensor:
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) # Chuyển sang ảnh xám
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY) # Nhị phân hóa ảnh
    inp = cv2.resize(bw, size) # Resize về kích thước đầu vào của model
    inp = inp.astype(np.float32) / 255.0 # Chuẩn hóa giá trị pixel về [0, 1]
    tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)  # Thêm chiều batch và channel: (1,1,H,W)
    return tensor

# Hàm thực hiện dự đoán bằng model PyTorch
def torch_predict(model: torch.nn.Module, image: np.ndarray, device: torch.device = torch.device('cpu')) -> np.ndarray:
    tensor = torch_process_image(image).to(device) # Tiền xử lý và chuyển tensor lên device (CPU/GPU)
    with torch.no_grad(): # Không tính toán gradient để tăng tốc
        out = model(tensor) # Đưa qua model
        if out.dim() == 4:
            out = out.view(out.size(0), -1)
        # Dùng softmax để chuyển output thành xác suất
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    # Trả về 3 lớp có xác suất cao nhất
    return np.argsort(probs)[-3:][::-1]

# Hàm kiểm tra cử chỉ "mở bàn tay" (5 ngón tay duỗi thẳng)
def fingers_open(landmarks: List[List[int]]):
    if len(landmarks) < 21: # Phải có đủ 21 điểm mốc
        return False
    # Kiểm tra vị trí của các đầu ngón tay so với các khớp gần nó
    return all([
        landmarks[4][1] < landmarks[3][1],     # Ngón cái
        landmarks[8][2] < landmarks[6][2],     # Ngón trỏ
        landmarks[12][2] < landmarks[10][2],   # Ngón giữa
        landmarks[16][2] < landmarks[14][2],   # Ngón áp út
        landmarks[20][2] < landmarks[18][2]    # Ngón út
    ])


# --------------------------
# Lớp chứa trạng thái của game (State container)
# --------------------------

@dataclass
class GameState:
    # Lớp này dùng để chứa tất cả các biến trạng thái của game
    # giúp cho việc quản lý dễ dàng hơn thay vì dùng biến toàn cục.
    width: int = WIDTH
    height: int = HEIGHT
    canvas: np.ndarray = field(default_factory=lambda: np.zeros((HEIGHT, WIDTH, 3), np.uint8)) # Vùng canvas để vẽ
    logo_img: Optional[np.ndarray] = None
    icons: List = field(default_factory=list) # Danh sách các icon đã load
    tool_color: Tuple[int, int, int] = (0, 255, 255) # Màu bút vẽ
    brush_size: int = 25 # Kích thước bút vẽ
    xp: int = 0 # Tọa độ x trước đó của bút
    yp: int = 0 # Tọa độ y trước đó của bút
    is_saved: bool = False # Cờ báo đã nộp bài vẽ chưa
    is_draw: bool = False # Cờ báo đang trong chế độ vẽ
    is_spam: bool = True # Cờ để chuyển sang vật thể tiếp theo
    is_play: bool = False # Cờ báo game đang bắt đầu
    start_time: float = 0 # Thời điểm bắt đầu màn chơi
    total_time: int = 60 # Tổng thời gian một màn
    display_time: float = 0 # Thời điểm hiển thị icon kết quả
    result_icon = None # Icon kết quả (đúng/sai)
    score: int = 0
    combo: int = 0
    draw_count: int = 0 # Số lần vẽ thành công
    frame_count: int = 0 # Đếm số frame đã xử lý
    target: str = '' # Tên vật thể cần vẽ
    target_id: int = 0 # ID của vật thể cần vẽ
    emo_list: List[str] = field(default_factory=list) # Danh sách các vật thể trong màn chơi
    emo_id: np.ndarray = field(default_factory=lambda: np.array([]))
    emo_pos: List[int] = field(default_factory=list)

    # Hàm để reset trạng thái game về ban đầu
    def reset(self):
        self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
        self.xp = self.yp = 0
        self.is_saved = False
        self.is_draw = False
        self.is_spam = True
        self.is_play = False
        self.start_time = 0
        self.display_time = 0
        self.result_icon = None
        self.score = 0
        self.combo = 0
        self.draw_count = 0
        self.frame_count = 0
        self.target = ''
        self.target_id = 0


# --------------------------
# Hàm chính (Main)
# --------------------------

def main():
    # --- KHỞI TẠO ---
    # Mở camera, thiết lập chiều rộng, cao
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    cap.set(10, 150) # Độ sáng

    # Tải các tài nguyên (logo, icons)
    logo = load_logo(LOGO_PATH)
    icons = load_icons(ICON_FOLDER)

    # Tải model PyTorch
    if not MODEL_PATH.exists():
        print("Không tìm thấy model:", MODEL_PATH)
        return
    device = torch.device('cpu') # Sử dụng CPU để dự đoán
    model = VGG_Small(num_classes=len(CLASSES_VN))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    # Tải trọng số đã huấn luyện vào model
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # Chuyển model sang chế độ đánh giá (không huấn luyện)

    # Khởi tạo đối tượng trạng thái game
    gs = GameState()
    gs.logo_img = logo
    gs.icons = icons

    # Tạo danh sách các vật thể ngẫu nhiên cho màn chơi
    GAME_SIZE = 40
    if len(CLASSES_VN) == 0:
        print("Không có lớp nào được định nghĩa")
        return
    gs.emo_id = np.random.choice(len(CLASSES_VN), GAME_SIZE)
    gs.emo_list = [CLASSES_VN[i] for i in gs.emo_id]
    gs.emo_pos = [i * 12 for i in range(GAME_SIZE)]

    # Khởi tạo MediaPipe Hands
    mp_hands = mp.solutions.hands

    try:
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            # --- VÒNG LẶP CHÍNH CỦA GAME ---
            while True:
                ret, frame = cap.read() # Đọc một frame từ camera
                if not ret:
                    print("Không thể đọc frame từ camera")
                    break
                frame = cv2.flip(frame, 1) # Lật frame theo chiều ngang

                # Đảm bảo canvas có cùng kích thước với frame từ camera
                if gs.canvas.shape != frame.shape:
                    gs.canvas = cv2.resize(gs.canvas, (frame.shape[1], frame.shape[0]))

                raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Chuyển sang RGB cho MediaPipe

                # --- VẼ GIAO DIỆN NGƯỜI DÙNG (UI) ---
                # Vẽ khung vẽ ở giữa màn hình
                start_point = (CENTER[0] - BOX_RANGE, CENTER[1] - BOX_RANGE)
                end_point = (CENTER[0] + BOX_RANGE, CENTER[1] + BOX_RANGE)
                cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=5)

                # Chọn vật thể cần vẽ tiếp theo
                if gs.is_spam and gs.emo_list:
                    idx = gs.frame_count % len(gs.emo_list)
                    gs.target = gs.emo_list[idx]
                    gs.target_id = gs.emo_id[idx]
                    gs.is_spam = False
                
                # Hiển thị tên vật thể cần vẽ
                frame = put_text_unicode(frame, gs.target, (CENTER[0] - 150, (CENTER[1] - BOX_RANGE) // 5), FONT_PATH, FONT_SIZE, (0, 0, 255))
                
                # --- LOGIC TRẠNG THÁI GAME ---
                # Bắt đầu game (khi nhấn 'p')
                if gs.is_play:
                    gs.start_time = time.time()
                    gs.combo = 0
                    gs.score = 0
                    gs.is_play = False

                # Xử lý khi game đang chạy
                if gs.start_time != 0:
                    elapsed = int(time.time() - gs.start_time)
                    if elapsed < gs.total_time: # Nếu vẫn còn thời gian
                        # Hiển thị điểm, combo, thời gian còn lại
                        cv2.putText(frame, f"Score: {gs.score}", (WIDTH - 225, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame, f"x{gs.combo} Combo", (WIDTH - 225, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv2.circle(frame, (50, 40), 30, (0, 0, 255), 5)
                        cv2.putText(frame, f"{int(gs.total_time - elapsed)}", (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
                    elif elapsed < gs.total_time + 10: # Hết giờ, hiển thị tổng kết
                        cv2.putText(frame, f"Score: {gs.score}", (WIDTH // 2 - 150, HEIGHT // 2 - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                        cv2.putText(frame, f"Draw: {gs.draw_count}", (WIDTH // 2 - 150, HEIGHT // 2 + 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                    else: # Sau khi hiển thị tổng kết, reset game
                        gs.reset()

                # --- XỬ LÝ NHẬN DIỆN BÀN TAY ---
                # Chỉ xử lý sau mỗi FRAME_SKIP frame để tiết kiệm tài nguyên
                if gs.frame_count % FRAME_SKIP == 0:
                    results = hands.process(raw)
                    landmarks = []
                    if results and results.multi_hand_landmarks:
                        # Trích xuất tọa độ các điểm mốc
                        for h in results.multi_hand_landmarks:
                            for id, lm in enumerate(h.landmark):
                                h_f, w_f, _ = frame.shape
                                cx, cy = int(lm.x * w_f), int(lm.y * h_f)
                                landmarks.append([id, cx, cy])

                    if landmarks:
                        # Lấy tọa độ đầu ngón trỏ và ngón giữa
                        x1, y1 = landmarks[8][1], landmarks[8][2]
                        x2, y2 = landmarks[12][1], landmarks[12][2]

                        # --- LOGIC CỬ CHỈ ---
                        # Cử chỉ 1: Nộp bài (mở cả bàn tay)
                        if fingers_open(landmarks) and gs.is_draw and not gs.is_saved:
                            # Cắt vùng ảnh đã vẽ từ canvas
                            y0 = max(0, CENTER[1] - BOX_RANGE)
                            y1b = min(gs.canvas.shape[0], CENTER[1] + BOX_RANGE)
                            x0 = max(0, CENTER[0] - BOX_RANGE)
                            x1b = min(gs.canvas.shape[1], CENTER[0] + BOX_RANGE)
                            box = gs.canvas[y0:y1b, x0:x1b]
                            
                            if box.size == 0:
                                class_label = np.array([-1]) # Nếu không vẽ gì
                            else:
                                # Dự đoán hình vẽ
                                class_label = torch_predict(model, box, device)
                            
                            gs.display_time = time.time()
                            gs.canvas = np.zeros((gs.height, gs.width, 3), np.uint8) # Xóa canvas
                            gs.is_saved = True
                            gs.is_draw = False
                            
                            # Kiểm tra kết quả
                            if class_label.size > 0 and gs.target_id in class_label: # Nếu đúng
                                num = np.random.randint(0, NUM_PER_CLASS)
                                if gs.icons:
                                    icon_index = gs.target_id * NUM_PER_CLASS + num
                                    icon_index = min(icon_index, len(gs.icons) - 1)
                                    gs.result_icon = gs.icons[icon_index] # Hiển thị icon đúng
                                else:
                                    gs.result_icon = None
                                gs.is_spam = True # Chuyển sang vật thể mới
                                gs.combo += 1
                                gs.draw_count += 1
                            else: # Nếu sai
                                if gs.icons and class_label.size > 0:
                                    icon_index = class_label[0] * NUM_PER_CLASS
                                    icon_index = min(icon_index, len(gs.icons) - 1)
                                    gs.result_icon = gs.icons[icon_index] # Hiển thị icon đã vẽ
                                else:
                                    gs.result_icon = None
                                gs.combo = 0 # Reset combo
                            gs.score += 100 * gs.combo # Cập nhật điểm

                        # Cử chỉ 2: Vẽ (chỉ giơ ngón trỏ)
                        elif landmarks[8][2] < landmarks[6][2]:
                            gs.is_saved = False
                            gs.is_draw = True
                            if gs.xp == 0 and gs.yp == 0: # Lấy điểm bắt đầu vẽ
                                gs.xp, gs.yp = x1, y1
                            # Vẽ một đường thẳng từ điểm cũ đến điểm mới
                            cv2.line(gs.canvas, (gs.xp, gs.yp), (x1, y1), gs.tool_color, gs.brush_size)
                            gs.xp, gs.yp = x1, y1 # Cập nhật điểm cũ

                # --- KẾT XUẤT HÌNH ẢNH ---
                # Dán logo
                if gs.logo_img is not None:
                    paste_image_safe(frame, gs.logo_img, (10, frame.shape[0] - gs.logo_img.shape[0] - 10))

                # Trộn canvas (hình vẽ) vào frame (video camera)
                gray = cv2.cvtColor(gs.canvas, cv2.COLOR_BGR2GRAY)
                _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV) # Tạo mask ngược
                inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
                
                # Đảm bảo mask và frame có cùng kích thước
                if inv.shape != frame.shape:
                    inv = cv2.resize(inv, (frame.shape[1], frame.shape[0]))

                print(f"Frame shape: {frame.shape}")
                print(f"Canvas shape: {gs.canvas.shape}")
                # 1. Get the height and width of the camera frame
                h, w, _ = frame.shape

                # 2. Resize the canvas to match the frame's dimensions
                resized_canvas = cv2.resize(gs.canvas, (w, h))

                # Dùng bitwise operations để "cắt" một lỗ trên frame và chèn hình vẽ vào
                frame = cv2.bitwise_and(frame, inv)
                # 3. Now perform the bitwise operation with the correctly sized images
                frame = cv2.bitwise_or(frame, resized_canvas)
                # frame = cv2.bitwise_or(frame, gs.canvas)

                # Hiển thị icon kết quả trong một khoảng thời gian ngắn (2 giây)
                if gs.result_icon is not None and (time.time() - gs.display_time) < 2:
                    paste_image_safe(frame, gs.result_icon, (CENTER[0] - EMO_SIZE // 2, CENTER[1] - EMO_SIZE // 2))

                # Hiển thị frame cuối cùng
                cv2.imshow('cam', frame)

                # --- XỬ LÝ ĐẦU VÀO TỪ BÀN PHÍM ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): # Nhấn 'q' để thoát
                    break
                if key == ord('n'): # Nhấn 'n' để qua vật thể tiếp theo
                    gs.is_spam = True
                if key == ord('p'): # Nhấn 'p' để bắt đầu chơi
                    gs.is_play = True
                if key == ord('c'): # Nhấn 'c' để xóa hình vẽ
                    gs.canvas = np.zeros_like(gs.canvas)

                gs.frame_count += 1

    finally:
        # Giải phóng tài nguyên khi kết thúc
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()