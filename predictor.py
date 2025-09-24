# predictor.py
import threading
import queue
import time
import numpy as np
import pygame
import cv2
import torch
from settings import IMG_SIZE, MASK_THRESHOLD, MODEL_PATH, CLASSES_VN
from settings import *
from ModelArchitect import VGG_Small # Đảm bảo file ModelArchitect.py ở cùng thư mục

class Predictor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.prediction_queue = queue.Queue(maxsize=1)
        self.latest_result = None
        self.lock = threading.Lock()
        self.running = False
        self.model = self._load_model()

    def _load_model(self):
        if not MODEL_PATH.exists():
            print(f"Lỗi: Không tìm thấy model tại {MODEL_PATH}")
            return None
        try:
            device = torch.device('cpu')
            model = VGG_Small(num_classes=len(CLASSES_VN))
            state_dict = torch.load(MODEL_PATH, map_location=device)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print("Model AI đã được tải thành công.")
            return model
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            return None

    def _preprocess_surface(self, surface: pygame.Surface) -> torch.Tensor:
        """
        Tiền xử lý pygame.Surface theo logic mới, bao gồm cả bước chuẩn hóa.
        """
        # 1. Chuyển pygame.Surface thành mảng numpy BGR của OpenCV
        view = pygame.surfarray.pixels3d(surface)
        img_bgr = np.transpose(view, (1, 0, 2))
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR) # surfarray là RGB, opencv cần BGR

        # 2. Áp dụng logic xử lý mới
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
        inp = cv2.resize(bw, IMG_SIZE).astype(np.float32) / 255.0  # Chuyển về dải [0, 1]

        # 3. Chuẩn hóa về dải [-1, 1]
        inp = (inp - 0.5) / 0.5

        # 4. (Tùy chọn) Lưu ảnh debug để kiểm tra
        # Dòng này sẽ tạo một file debug_input.png trong thư mục dự án mỗi khi bạn nộp bài.
        try:
            cv2.imwrite("debug_input.png", ((inp + 1) * 127.5).astype(np.uint8))
        except Exception as e:
            print(f"Lỗi khi lưu ảnh debug: {e}")

        # 5. Chuyển thành Tensor
        return torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    
    def run(self):
        self.running = True
        while self.running:
            try:
                surface_to_predict = self.prediction_queue.get(timeout=1)
                if surface_to_predict is None:
                    continue

                if self.model:
                    tensor = self._preprocess_surface(surface_to_predict)
                    with torch.no_grad():
                        out = self.model(tensor)
                        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                    
                    # Lấy 3 kết quả có xác suất cao nhất
                    top3_indices = np.argsort(probs)[-3:][::-1]
                    
                    # THAY ĐỔI: Tạo một danh sách kết quả chi tiết hơn
                    top3_results = []
                    for i in top3_indices:
                        class_name = CLASSES_VN[i]
                        probability = probs[i]
                        top3_results.append((class_name, probability))
                    
                    with self.lock:
                        # Lưu danh sách kết quả chi tiết này
                        self.latest_result = top3_results
                
                self.prediction_queue.task_done()

            except queue.Empty:
                continue
        print("Predictor đã dừng.")

    def stop(self):
        self.running = False
        self.prediction_queue.put(None) # Gửi tín hiệu để bỏ block .get()

    def submit_for_prediction(self, surface: pygame.Surface):
        if self.prediction_queue.empty():
            self.latest_result = None # Xóa kết quả cũ
            self.prediction_queue.put(surface)
            return True
        return False # Hàng đợi đang bận, không nhận thêm

    def get_latest_result(self):
        with self.lock:
            return self.latest_result