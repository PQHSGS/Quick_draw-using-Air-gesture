# hand_tracker.py
import cv2
import mediapipe as mp
import threading
import time

class HandTracker(threading.Thread):
    # THAY ĐỔI: Không nhận cap_index nữa, mà nhận đối tượng webcam_stream
    def __init__(self, webcam_stream):
        super().__init__()
        self.daemon = True
        self.webcam_stream = webcam_stream # Lưu lại stream
        self.landmarks = []
        self.running = False
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            while self.running:
                # THAY ĐỔI: Lấy frame từ stream thay vì tự đọc
                frame = self.webcam_stream.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Không cần flip nữa vì stream đã làm
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                temp_landmarks = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        h, w, _ = frame.shape
                        for id, lm in enumerate(hand_landmarks.landmark):
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            temp_landmarks.append([id, cx, cy])
                
                with self.lock:
                    self.landmarks = temp_landmarks
                
                # Có thể xóa sleep ở đây hoặc giữ lại để giảm tải CPU cho thread này
                time.sleep(0.01)
        print("Hand tracker đã dừng.")

    def stop(self): self.running = False
    def get_landmarks(self):
        with self.lock: return self.landmarks