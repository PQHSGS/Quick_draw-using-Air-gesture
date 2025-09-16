# game.py (fixed: paste_result_icon present, removed unused load_logo call)
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ModelArchitect import VGG_Small
from HandTrack import HandDetector

# ---- minimal hyperparams ----
DETECTION_SKIP = 1
PREDICT_COOLDOWN = 1.5
MASK_THRESHOLD = 50
BRUSH_SIZE = 25
RESULT_DISPLAY = 2.0

# ---- config ----
WIDTH, HEIGHT = 1200, 800
CENTER = (WIDTH // 2, HEIGHT // 2 - 150)
BOX_RANGE = 225
EMO_SIZE = 300
NUM_PER_CLASS = 3
FONT_PATH = "arial.ttf"
FONT_SIZE = 45
ICON_FOLDER = Path("icon_v2")
MODEL_PATH = Path("vgg.pt")

CLASSES_VN = np.array([
    'Quả táo','Quả chuối','Bánh trung thu','Con tàu','Bánh cá','Mặt nạ',
    'Bông hoa','Đèn lồng','Con lân','Ông trăng','Quả lê','Quả dứa','Thỏ ngọc',
    'Đèn ông sao','Quả dâu tây','Cây thần','Quả dưa hấu'
])

class Phase(Enum):
    IDLE = auto(); DRAWING = auto(); SUBMITTED = auto(); PLAYING = auto()

@dataclass
class GameState:
    phase: Phase = Phase.IDLE
    canvas: np.ndarray = field(default_factory=lambda: np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    xp: int = 0; yp: int = 0
    last_predict_ts: float = 0.0
    last_result_ts: float = 0.0
    is_drawing: bool = False
    score: int = 0; combo: int = 0; draw_count: int = 0
    frame_count: int = 0
    target_id: int = 0; target_name: str = ''
    emo_list: List[str] = field(default_factory=list)
    emo_id: np.ndarray = field(default_factory=lambda: np.array([]))
    emo_pos: List[int] = field(default_factory=list)
    result_icon = None

    def reset_canvas(self, frame_shape=None):
        if frame_shape is None:
            self.canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        else:
            h, w = frame_shape[:2]; self.canvas = np.zeros((h, w, 3), np.uint8)
        self.xp = self.yp = 0; self.phase = Phase.IDLE

# ---- helpers ----
def load_icons(folder: Path, emo_size=EMO_SIZE):
    if not folder.exists(): return []
    out = []
    for f in sorted(folder.rglob("*")):
        if not f.is_file(): continue
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if im is None: continue
        im = cv2.resize(im, (emo_size, emo_size))
        if im.ndim == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        out.append((im[:, :, :3], im[:, :, 3] if im.shape[2] == 4 else None))
    return out

def paste_result_icon(base: np.ndarray, overlay, topleft: Tuple[int, int]):
    if overlay is None: return
    rgb, alpha = overlay if isinstance(overlay, tuple) else (overlay, None)
    h, w = rgb.shape[:2]; x, y = topleft
    x1, y1 = max(0, x), max(0, y); x2, y2 = min(base.shape[1], x + w), min(base.shape[0], y + h)
    if x1 >= x2 or y1 >= y2: return
    ox1, oy1 = x1 - x, y1 - y
    patch = rgb[oy1:oy1 + (y2 - y1), ox1:ox1 + (x2 - x1)]
    roi = base[y1:y2, x1:x2].astype(np.float32)
    if alpha is not None:
        a = (alpha[oy1:oy1 + (y2 - y1), ox1:ox1 + (x2 - x1)] / 255.0)[..., None]
        base[y1:y2, x1:x2] = (a * patch + (1 - a) * roi).astype(np.uint8)
    else:
        base[y1:y2, x1:x2] = cv2.addWeighted(roi.astype(np.uint8), 0.3, patch, 0.7, 0)

def put_text_unicode(img, text, pos, font_path=FONT_PATH, font_size=FONT_SIZE, color=(255,255,255)):
    try:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil); font = ImageFont.truetype(font_path, font_size)
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size/30.0, color, 2); return img

def pick_new_target(gs: GameState):
    if not gs.emo_list:
        gs.target_id = -1; gs.target_name = ""; return
    i = np.random.randint(len(gs.emo_list))
    gs.target_name = gs.emo_list[i]; gs.target_id = int(gs.emo_id[i])

def check_draw(x, y):
    return (CENTER[0]-BOX_RANGE) <= x <= (CENTER[0]+BOX_RANGE) and (CENTER[1]-BOX_RANGE) <= y <= (CENTER[1]+BOX_RANGE)

# ---- torch utils ----
def torch_process_image(canvas: np.ndarray, size=(32,32)):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
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
    for r, idx in enumerate(inds, 1):
        print(f"  {r}. {CLASSES_VN[idx]} ({probs[idx]:.3f})", end=' ')
    print()
    return inds

def fingers_state_from_detector(detector) -> str:
    fingers = detector.fingersUp()
    if not fingers or len(fingers) < 5: return 'none'
    s = sum(fingers)
    if s == 5: return 'submit'
    if fingers[1] == 1 and fingers[2] == 0: return 'draw'
    if fingers[1] == 1 and fingers[2] == 1: return 'neutral'
    return 'none'

# ---- predict / render ----
def attempt_predict(gs: GameState, model, frame, now, icons):
    y0 = max(0, CENTER[1] - BOX_RANGE); y1b = min(gs.canvas.shape[0], CENTER[1] + BOX_RANGE)
    x0 = max(0, CENTER[0] - BOX_RANGE); x1b = min(gs.canvas.shape[1], CENTER[0] + BOX_RANGE)
    box = gs.canvas[y0:y1b, x0:x1b]
    class_label = np.array([-1]) if box.size == 0 or box.shape[0] < 4 or box.shape[1] < 4 else torch_predict(model, box)
    success = class_label.size > 0 and gs.target_id in class_label
    chosen_icon = None
    if success and icons:
        base_idx = gs.target_id * NUM_PER_CLASS
        if base_idx < len(icons):
            chosen_icon = icons[min(base_idx + np.random.randint(0, NUM_PER_CLASS), len(icons)-1)]
    gs.reset_canvas(frame.shape)
    gs.last_predict_ts = now
    gs.result_icon = chosen_icon if success else None
    gs.last_result_ts = now
    gs.xp = gs.yp = 0
    if success:
        gs.combo += 1; gs.draw_count += 1; gs.score += 100 * gs.combo
    else:
        gs.combo = 0
    gs.phase = Phase.SUBMITTED
    pick_new_target(gs)

def blend_canvas(frame, gs):
    if gs.canvas.dtype != frame.dtype: gs.canvas = gs.canvas.astype(frame.dtype)
    gray = cv2.cvtColor(gs.canvas, cv2.COLOR_BGR2GRAY)
    if not np.any(gray): return frame
    _, mask_inv = cv2.threshold(gray, MASK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    mask3 = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    if mask3.shape[:2] != frame.shape[:2]:
        mask3 = cv2.resize(mask3, (frame.shape[1], frame.shape[0])); gs.canvas = cv2.resize(gs.canvas, (frame.shape[1], frame.shape[0]))
    return cv2.bitwise_or(cv2.bitwise_and(frame, mask3), gs.canvas)

# ---- main ----
def build_model(path: Path, device=torch.device('cpu')):
    if not path.exists(): raise FileNotFoundError(path)
    model = VGG_Small(num_classes=len(CLASSES_VN))
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd.get('state_dict', sd)); model.to(device); model.eval()
    return model

def main():
    cap = cv2.VideoCapture(0); cap.set(3, WIDTH); cap.set(4, HEIGHT)
    icons = load_icons(ICON_FOLDER)
    device = torch.device('cpu'); model = build_model(MODEL_PATH, device)
    gs = GameState()
    GAME_SIZE = 40
    gs.emo_id = np.random.choice(len(CLASSES_VN), GAME_SIZE)
    gs.emo_list = [CLASSES_VN[i] for i in gs.emo_id]; gs.emo_pos = [i * 12 for i in range(GAME_SIZE)]
    pick_new_target(gs)
    detector = HandDetector(maxHands=1, detectionCon=0.7, trackCon=0.7)
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            if gs.canvas.shape[:2] != frame.shape[:2]: gs.reset_canvas(frame.shape)
            if gs.frame_count % DETECTION_SKIP == 0:
                detector.findHands(frame, draw=False); landmarks = detector.findPosition(frame)
            else:
                landmarks = []
            cv2.rectangle(frame, (CENTER[0]-BOX_RANGE, CENTER[1]-BOX_RANGE), (CENTER[0]+BOX_RANGE, CENTER[1]+BOX_RANGE), (0,0,0), 5)
            frame = put_text_unicode(frame, gs.target_name, (CENTER[0]-20, (CENTER[1]-BOX_RANGE)//5))
            status = 'none'
            if landmarks:
                x1, y1 = landmarks[8][1], landmarks[8][2]
                status = fingers_state_from_detector(detector); now = time.time()
                if status == 'draw':
                    gs.phase = Phase.DRAWING
                    if gs.xp == 0 and gs.yp == 0: gs.xp, gs.yp = x1, y1
                    cv2.line(gs.canvas, (gs.xp, gs.yp), (x1, y1), (0,255,255), BRUSH_SIZE, cv2.FILLED)
                    gs.xp, gs.yp = x1, y1
                    gs.is_drawing = check_draw(x1, y1) or gs.is_drawing
                elif status == 'submit':
                    if gs.is_drawing and (time.time() - gs.last_predict_ts) >= PREDICT_COOLDOWN:
                        attempt_predict(gs, model, frame, time.time(), icons); gs.is_drawing = False
                else:
                    gs.phase = Phase.IDLE; gs.xp = gs.yp = 0
            frame = blend_canvas(frame, gs)
            if gs.result_icon is not None and (time.time() - gs.last_result_ts) < RESULT_DISPLAY:
                paste_result_icon(frame, gs.result_icon, (CENTER[0] - EMO_SIZE // 2, CENTER[1] - EMO_SIZE // 2))
            else:
                gs.result_icon = None
            cv2.imshow('cam', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            if k == ord('c'): gs.reset_canvas(frame.shape)
            if k == ord('p'): gs.phase = Phase.PLAYING
            gs.frame_count += 1
    finally:
        cap.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
