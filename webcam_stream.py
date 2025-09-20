# webcam_stream.py
import cv2
import threading
import time
from settings import WIDTH, HEIGHT

class WebcamStream(threading.Thread):
    """
    Một thread chuyên dụng chỉ để đọc frame từ camera với tốc độ nhanh nhất có thể.
    """
    def __init__(self, cap_index=0):
        super().__init__()
        self.daemon = True
        self.cap = cv2.VideoCapture(cap_index)
        if not self.cap.isOpened():
            raise IOError(f"Không thể mở camera {cap_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        
        # Đọc frame đầu tiên
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise IOError("Không thể đọc frame đầu tiên từ camera.")

        self.lock = threading.Lock()
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            # Liên tục đọc frame mới mà không bị chặn bởi các tác vụ khác
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                # Ngủ một chút nếu đọc thất bại để tránh vắt kiệt CPU
                time.sleep(0.01)

    def get_frame(self):
        """Trả về frame mới nhất một cách an toàn."""
        with self.lock:
            # Lật frame ở đây để tất cả các consumer đều nhận được frame đã lật
            return cv2.flip(self.frame, 1) if self.frame is not None else None

    def stop(self):
        self.running = False
        time.sleep(0.1) # Đợi một chút để vòng lặp kết thúc
        self.cap.release()
        print("Webcam stream đã dừng.")