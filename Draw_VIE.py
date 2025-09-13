"""
Refactored Quick Draw script.
Improvements:
- Structured flow using GameState class and helper functions.
- Safe image overlay that supports alpha masks and clamps edges.
- Fixed tool placement bug and index OOB issues.
- Reduced repeated cv2.waitKey calls.
- Frame skipping configurable.
- Robust resource loading with clear error messages.
- Clear separation of concerns: I/O, processing, UI.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import mediapipe as mp
import numpy as np
import torch
from ModelArchitect import VGG_Small
from PIL import Image, ImageDraw, ImageFont

# --------------------------
# Configuration
# --------------------------
FRAME_SKIP = 2  # process every N-th frame
WIDTH = 1200
HEIGHT = 800
OFFSET = 20
LIMIT = 600
CENTER = (340, 300)
BOX_RANGE = 225
IMG_SIZE = 120
EMO_SIZE = 300
NUM_PER_CLASS = 3
FONT_PATH = "arial.ttf"
FONT_SIZE = 45
TOOLS_FOLDER = Path("tools")
ICON_FOLDER = Path("icon_v2")
MODEL_PATH = Path("vgg.pt")
LOGO_PATH = Path("vme.jpg")

CLASSES_VN = np.array([
    'Quả táo', 'Quả chuối', 'Bánh trung thu', 'Con tàu', 'Bánh cá', 'Mặt nạ',
    'Bông hoa', 'Đèn lồng', 'Con lân', 'Ông trăng', 'Quả lê', 'Quả dứa', 'Thỏ ngọc',
    'Đèn ông sao', 'Quả dâu tây', 'Cây thần', 'Quả dưa hấu'
])

# --------------------------
# Helpers
# --------------------------

def load_tools(folder: Path, size: int = IMG_SIZE) -> List[np.ndarray]:
    if not folder.exists():
        raise FileNotFoundError(f"Tools folder not found: {folder}")
    imgs = []
    for f in sorted(folder.iterdir()):
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img = cv2.resize(img, (size, size))
        imgs.append(img)
    if not imgs:
        raise RuntimeError("No tool images loaded.")
    return imgs


def load_logo(path: Path, size: Tuple[int, int] = (70, 70)) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return cv2.resize(img, size)


def load_icons(folder: Path, emo_size: int = EMO_SIZE) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Load icons. Return list of (rgb_image, alpha_mask_or_None).
    alpha_mask is single-channel uint8 (0..255) when available.
    """
    if not folder.exists():
        return []
    icons = []
    for cls in sorted(folder.iterdir()):
        if not cls.is_dir():
            continue
        for f in sorted(cls.iterdir()):
            im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            im = cv2.resize(im, (emo_size, emo_size))
            # ensure we have at least 2 dims
            if im.ndim == 3 and im.shape[2] == 4:
                rgb = im[:, :, :3]
                alpha = im[:, :, 3]
                icons.append((rgb, alpha))
            else:
                # if grayscale, convert to BGR
                if im.ndim == 2:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                icons.append((im, None))
    return icons


def paste_image_safe(base: np.ndarray, overlay, topleft: Tuple[int, int]):
    """Paste overlay onto base at topleft.

    overlay may be either:
      - an ndarray with shape (h, w, 4) (RGBA)
      - an ndarray with shape (h, w, 3) (BGR)
      - a tuple (rgb_ndarray, alpha_single_channel) where alpha is shape (h,w)
    Function clamps edges and blends using alpha if present.
    """
    if overlay is None:
        return

    # normalize overlay to (h,w,4) when alpha exists, else (h,w,3)
    if isinstance(overlay, tuple):
        rgb, alpha = overlay
        if alpha is not None:
            # make RGBA
            alpha_ch = alpha if alpha.ndim == 2 else alpha[:, :, 0]
            overlay_arr = np.dstack([rgb, alpha_ch])
        else:
            overlay_arr = rgb
    else:
        overlay_arr = overlay

    if overlay_arr.ndim < 3:
        return

    bh, bw = base.shape[:2]
    th, tw = overlay_arr.shape[:2]
    x, y = topleft

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + tw)
    y2 = min(bh, y + th)
    if x1 >= x2 or y1 >= y2:
        return

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    roi = base[y1:y2, x1:x2]
    patch = overlay_arr[oy1:oy2, ox1:ox2]

    # If patch has alpha channel
    if patch.shape[2] == 4:
        alpha = patch[:, :, 3].astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        rgb_patch = patch[:, :, :3].astype(np.float32)
        roi[:] = (alpha * rgb_patch + (1 - alpha) * roi.astype(np.float32)).astype(np.uint8)
    else:
        # no alpha: lightweight blend to avoid full overwrite
        base[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.3, patch[:, :, :3], 0.7, 0)


def put_text_unicode(img: np.ndarray, text: str, position: Tuple[int, int], font_path: str = FONT_PATH,
                     font_size: int = FONT_SIZE, color=(255, 255, 255)) -> np.ndarray:
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, font_size)
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # fallback to OpenCV text
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color, 2)
        return img


def torch_process_image(canvas: np.ndarray):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    inp = cv2.resize(bw, (32, 32))
    inp = inp.reshape(1, 32, 32, 1)
    return inp


def torch_predict(model, image: np.ndarray):
    p = model.predict(torch_process_image(image), verbose=0)[0]
    return np.argsort(p)[-3:][::-1]


def fingers_open(landmarks: List[List[int]]):
    if len(landmarks) < 21:
        return False
    return all([
        landmarks[4][1] < landmarks[3][1],
        landmarks[8][2] < landmarks[6][2],
        landmarks[12][2] < landmarks[10][2],
        landmarks[16][2] < landmarks[14][2],
        landmarks[20][2] < landmarks[18][2]
    ])


# --------------------------
# State container
# --------------------------

@dataclass
class GameState:
    width: int = WIDTH
    height: int = HEIGHT
    canvas: np.ndarray = field(default_factory=lambda: np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    eraser_img: Optional[np.ndarray] = None
    pen_img: Optional[np.ndarray] = None
    logo_img: Optional[np.ndarray] = None
    icons: List = field(default_factory=list)
    tool_color: Tuple[int, int, int] = (0, 255, 255)
    brush_size: int = 25
    xp: int = 0
    yp: int = 0
    is_saved: bool = False
    is_draw: bool = False
    is_spam: bool = True
    is_play: bool = False
    start_time: float = 0
    total_time: int = 60
    display_time: float = 0
    result_icon = None
    score: int = 0
    combo: int = 0
    draw_count: int = 0
    frame_count: int = 0
    target: str = ''
    target_id: int = 0
    emo_list: List[str] = field(default_factory=list)
    emo_id: np.ndarray = field(default_factory=lambda: np.array([]))
    emo_pos: List[int] = field(default_factory=list)

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
# Main
# --------------------------


def main():
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    cap.set(10, 150)

    # resources
    try:
        tools = load_tools(TOOLS_FOLDER)
    except Exception as e:
        print("Error loading tools:", e)
        return
    logo = load_logo(LOGO_PATH)
    icons = load_icons(ICON_FOLDER)

    if not MODEL_PATH.exists():
        print("Model not found:", MODEL_PATH)
        return
    model = VGG_Small(num_classes=len(CLASSES_VN))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    gs = GameState()
    gs.eraser_img = tools[0]
    gs.pen_img = tools[1] if len(tools) > 1 else tools[0]
    gs.logo_img = logo
    gs.icons = icons
    # generate emo list
    GAME_SIZE = 40
    if len(CLASSES_VN) == 0:
        print("No classes defined")
        return
    gs.emo_id = np.random.choice(len(CLASSES_VN), GAME_SIZE)
    gs.emo_list = [CLASSES_VN[i] for i in gs.emo_id]
    gs.emo_pos = [i * 12 for i in range(GAME_SIZE)]

    mp_hands = mp.solutions.hands

    try:
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera frame not available")
                    break
                frame = cv2.flip(frame, 1)

                # Ensure canvas matches incoming frame shape to avoid bitwise errors
                if gs.canvas.shape != frame.shape:
                    gs.canvas = cv2.resize(gs.canvas, (frame.shape[1], frame.shape[0]))

                raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # draw interface rectangles
                start_point = (CENTER[0] - BOX_RANGE, CENTER[1] - BOX_RANGE)
                end_point = (CENTER[0] + BOX_RANGE, CENTER[1] + BOX_RANGE)
                cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=5)

                # tool boxes
                top_left_eraser = (LIMIT + OFFSET, HEIGHT // 2 - OFFSET * 3 - IMG_SIZE)
                top_left_pen = (LIMIT + OFFSET, HEIGHT // 2 + OFFSET * 3)
                cv2.rectangle(frame, (LIMIT + OFFSET, HEIGHT // 2 - OFFSET * 3 - IMG_SIZE), (LIMIT + OFFSET + IMG_SIZE, HEIGHT // 2 - OFFSET * 3), color=(0, 0, 0), thickness=5)
                cv2.rectangle(frame, (LIMIT + OFFSET, HEIGHT // 2 + OFFSET * 3), (LIMIT + OFFSET + IMG_SIZE, HEIGHT // 2 + OFFSET * 3 + IMG_SIZE), color=(0, 0, 0), thickness=5)

                # handle target selection
                if gs.is_spam and gs.emo_list:
                    idx = gs.frame_count % len(gs.emo_list)
                    gs.target = gs.emo_list[idx]
                    gs.target_id = gs.emo_id[idx]
                    gs.is_spam = False

                frame = put_text_unicode(frame, gs.target, (CENTER[0] - gs.emo_pos[0], (CENTER[1] - BOX_RANGE) // 5), FONT_PATH, FONT_SIZE, (0, 0, 255))

                # start timer when play pressed
                if gs.is_play:
                    gs.start_time = time.time()
                    gs.combo = 0
                    gs.score = 0
                    gs.is_play = False

                if gs.start_time != 0:
                    elapsed = int(time.time() - gs.start_time)
                    if elapsed < gs.total_time:
                        cv2.putText(frame, f"Score: {gs.score}", (WIDTH - 225, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame, f"x{gs.combo} Combo", (WIDTH - 225, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv2.circle(frame, (50, 40), 30, (0, 0, 255), 5)
                        cv2.putText(frame, f"{int(gs.total_time - elapsed)}", (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
                    elif elapsed < gs.total_time + 10:
                        cv2.putText(frame, f"Score: {gs.score}", (WIDTH // 2 - 150, HEIGHT // 2 - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                        cv2.putText(frame, f"Draw: {gs.draw_count}", (WIDTH // 2 - 150, HEIGHT // 2 + 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                    else:
                        gs.reset()

                # Process every FRAME_SKIP-th frame for Mediapipe
                if gs.frame_count % FRAME_SKIP == 0:
                    results = hands.process(raw)
                    landmarks = []
                    if results and results.multi_hand_landmarks:
                        for h in results.multi_hand_landmarks:
                            for id, lm in enumerate(h.landmark):
                                h_f, w_f, _ = frame.shape
                                cx, cy = int(lm.x * w_f), int(lm.y * h_f)
                                landmarks.append([id, cx, cy])

                    if landmarks:
                        x1, y1 = landmarks[8][1], landmarks[8][2]
                        x2, y2 = landmarks[12][1], landmarks[12][2]

                        # submit gesture (open hand)
                        if fingers_open(landmarks) and gs.is_draw and not gs.is_saved:
                            # clamp the box region inside canvas to avoid negative indexes
                            y0 = max(0, CENTER[1] - BOX_RANGE)
                            y1b = min(gs.canvas.shape[0], CENTER[1] + BOX_RANGE)
                            x0 = max(0, CENTER[0] - BOX_RANGE)
                            x1b = min(gs.canvas.shape[1], CENTER[0] + BOX_RANGE)
                            box = gs.canvas[y0:y1b, x0:x1b]
                            class_label = torch_predict(model, box)
                            gs.display_time = time.time()
                            gs.canvas = np.zeros((gs.height, gs.width, 3), np.uint8)
                            gs.is_saved = True
                            gs.is_draw = False
                            if gs.target_id in class_label:
                                num = np.random.randint(0, NUM_PER_CLASS)
                                if gs.icons:
                                    icon_index = gs.target_id * NUM_PER_CLASS + num
                                    icon_index = min(icon_index, len(gs.icons)-1)
                                    gs.result_icon = gs.icons[icon_index]
                                else:
                                    gs.result_icon = None
                                gs.is_spam = True
                                gs.combo += 1
                                gs.draw_count += 1
                            else:
                                if gs.icons:
                                    icon_index = class_label[0] * NUM_PER_CLASS
                                    icon_index = min(icon_index, len(gs.icons)-1)
                                    gs.result_icon = gs.icons[icon_index]
                                else:
                                    gs.result_icon = None
                                gs.combo = 0
                            gs.score += 100 * gs.combo

                        # tool change area
                        elif landmarks[8][2] < landmarks[6][2] and landmarks[12][2] < landmarks[10][2]:
                            gs.xp, gs.yp = 0, 0
                            # check if in eraser box
                            pen_box_top = HEIGHT // 2 + OFFSET * 3
                            eraser_box_top = HEIGHT // 2 - OFFSET * 3 - IMG_SIZE
                            if LIMIT < x1:
                                if eraser_box_top <= y1 <= eraser_box_top + IMG_SIZE:
                                    gs.tool_color = (0, 0, 0)
                                    gs.brush_size = 50
                                    cv2.rectangle(frame, (LIMIT + OFFSET, eraser_box_top), (LIMIT + OFFSET + IMG_SIZE, eraser_box_top + IMG_SIZE), (0, 255, 0), 5)
                                elif pen_box_top <= y1 <= pen_box_top + IMG_SIZE:
                                    gs.tool_color = (0, 255, 255)
                                    gs.brush_size = 25
                                    cv2.rectangle(frame, (LIMIT + OFFSET, pen_box_top), (LIMIT + OFFSET + IMG_SIZE, pen_box_top + IMG_SIZE), (0, 255, 0), 5)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), gs.tool_color, cv2.FILLED)

                        # drawing
                        elif landmarks[8][2] < landmarks[6][2]:
                            gs.is_saved = False
                            gs.is_draw = True
                            if gs.xp == 0 and gs.yp == 0:
                                gs.xp, gs.yp = x1, y1
                            cv2.line(gs.canvas, (gs.xp, gs.yp), (x1, y1), gs.tool_color, gs.brush_size, cv2.FILLED)
                            gs.xp, gs.yp = x1, y1

                # Place tools and logo safely
                paste_image_safe(frame, gs.eraser_img, top_left_eraser)
                paste_image_safe(frame, gs.pen_img, top_left_pen)
                if gs.logo_img is not None:
                    paste_image_safe(frame, gs.logo_img, (10, frame.shape[0] - gs.logo_img.shape[0] - 10))

                # blend canvas and frame
                gray = cv2.cvtColor(gs.canvas, cv2.COLOR_BGR2GRAY)
                _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
                inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
                if inv.shape != frame.shape:
                    inv = cv2.resize(inv, (frame.shape[1], frame.shape[0]))

                frame = cv2.bitwise_and(frame, inv)
                frame = cv2.bitwise_or(frame, gs.canvas)

                # show result icon for limited time
                if gs.result_icon is not None and (time.time() - gs.display_time) < 2:
                    paste_image_safe(frame, gs.result_icon, (CENTER[0] - EMO_SIZE // 2, CENTER[1] - EMO_SIZE // 2))

                cv2.imshow('cam', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('n'):
                    gs.is_spam = True
                if key == ord('p'):
                    gs.is_play = True

                gs.frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
