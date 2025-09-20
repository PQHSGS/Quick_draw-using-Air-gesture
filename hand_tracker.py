# hand_tracker.py
import cv2
import mediapipe as mp
import threading
import time
from settings import WIDTH, HEIGHT # MỚI: Import WIDTH và HEIGHT từ settings

class HandTracker(threading.Thread):
    def __init__(self, cap_index=0):
        super().__init__()
        self.daemon = True
        self.cap_index = cap_index
        self.landmarks = []
        self.running = False
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.cap_index)
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở camera {self.cap_index}")
            self.running = False
            return
            
        # MỚI: Thiết lập độ phân giải cho camera để khớp với Pygame
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        print(f"Camera được thiết lập với độ phân giải: {WIDTH}x{HEIGHT}")

        mp_hands = mp.solutions.hands
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            while self.running and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    # In ra kích thước frame thực tế để debug
                    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(f"Cảnh báo: Không đọc được frame. Kích thước thực tế của camera: {actual_w}x{actual_h}")
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                temp_landmarks = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for id, lm in enumerate(hand_landmarks.landmark):
                            # Bây giờ h, w sẽ là 1280x720, khớp với màn hình Pygame
                            h, w, _ = frame.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            temp_landmarks.append([id, cx, cy])
                
                with self.lock:
                    self.landmarks = temp_landmarks
                
                time.sleep(0.01)

        cap.release()
        print("Hand tracker đã dừng.")

    def stop(self):
        self.running = False

    def get_landmarks(self):
        with self.lock:
            return self.landmarks