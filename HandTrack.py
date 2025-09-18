# handtrack.py

# Mô tả: Một module độc lập sử dụng thư viện MediaPipe để phát hiện và theo dõi
#        các điểm mốc (landmarks) của bàn tay từ hình ảnh đầu vào.

import math
import cv2
import mediapipe as mp

class HandDetector:
    """
    Lớp HandDetector đóng gói các chức năng của MediaPipe Hands để dễ dàng
    tích hợp vào các ứng dụng khác.
    """
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        """
        Hàm khởi tạo của lớp HandDetector.

        Args:
            mode (bool): Chế độ tĩnh. Nếu là True, bộ phát hiện sẽ xử lý mỗi ảnh như một ảnh tĩnh,
                         phù hợp cho các ảnh không liên quan. Nếu là False, nó sẽ cố gắng theo dõi
                         bàn tay giữa các frame, phù hợp cho video.
            maxHands (int): Số lượng bàn tay tối đa có thể phát hiện.
            modelComplexity (int): Độ phức tạp của mô hình landmark (0 hoặc 1).
                                   Mô hình phức tạp hơn cho độ chính xác cao hơn nhưng chậm hơn.
            detectionCon (float): Ngưỡng tin cậy tối thiểu ([0.0, 1.0]) để coi một phát hiện là thành công.
            trackCon (float): Ngưỡng tin cậy tối thiểu ([0.0, 1.0]) để theo dõi bàn tay.
                              Nếu độ tin cậy giảm xuống dưới ngưỡng này, bộ phát hiện sẽ tự động
                              chạy lại trên frame tiếp theo.
        """
        # Lưu các tham số cấu hình vào thuộc tính của instance
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Khởi tạo giải pháp Hands của MediaPipe
        self.mpHands = mp.solutions.hands
        # Tạo một instance của bộ xử lý Hands với các cấu hình đã cho
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplex,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        # Khởi tạo công cụ vẽ của MediaPipe để trực quan hóa các landmark và kết nối
        self.mpDraw = mp.solutions.drawing_utils
        # Thuộc tính để lưu trữ kết quả phát hiện mới nhất
        self.results = None
        # Thuộc tính để lưu trữ danh sách các vị trí landmark đã được xử lý
        self.lmList = []

    def findHands(self, img, draw=False):
        """
        Tìm kiếm các bàn tay trong một khung hình (image).

        Args:
            img: Khung hình đầu vào (dưới dạng mảng NumPy từ OpenCV, thường là BGR).
            draw (bool): Nếu True, sẽ vẽ các landmark và kết nối lên hình ảnh.

        Returns:
            img: Khung hình (có thể đã được vẽ lên) sau khi xử lý.
        """
        # MediaPipe yêu cầu ảnh đầu vào ở định dạng RGB, trong khi OpenCV đọc ảnh là BGR.
        # Cần chuyển đổi không gian màu.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Xử lý ảnh RGB để tìm kiếm bàn tay và lưu kết quả vào self.results
        self.results = self.hands.process(imgRGB)

        # Kiểm tra xem có kết quả và có landmark của bàn tay nào được tìm thấy không
        if draw and self.results and self.results.multi_hand_landmarks:
            # Lặp qua từng bàn tay được phát hiện
            for handLms in self.results.multi_hand_landmarks:
                # Vẽ các điểm mốc (landmarks) và các đường nối giữa chúng lên ảnh gốc (img)
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        # Trả về ảnh đã được xử lý
        return img

    def findPosition(self, img, handNo=0):
        """
        Trích xuất vị trí (tọa độ pixel) của các landmark cho một bàn tay cụ thể.

        Args:
            img: Khung hình đầu vào để lấy kích thước (chiều rộng, chiều cao).
            handNo (int): Chỉ số của bàn tay cần lấy vị trí (mặc định là bàn tay đầu tiên, 0).

        Returns:
            list: Một danh sách các landmark, mỗi landmark là một list con có dạng [id, cx, cy].
                  Trả về danh sách rỗng nếu không có bàn tay nào được phát hiện.
        """
        # Xóa danh sách landmark cũ trước mỗi lần tìm kiếm mới
        self.lmList = []
        # Kiểm tra nếu không có kết quả hoặc không có landmark nào được phát hiện
        if not self.results or not self.results.multi_hand_landmarks:
            return self.lmList
        # Kiểm tra xem chỉ số bàn tay yêu cầu có hợp lệ không
        if handNo >= len(self.results.multi_hand_landmarks):
            return self.lmList

        # Lấy thông tin của bàn tay được chỉ định
        myHand = self.results.multi_hand_landmarks[handNo]
        # Lấy chiều cao (h), chiều rộng (w) của ảnh
        h, w, _ = img.shape
        # Lặp qua từng landmark (lm) trong bàn tay, cùng với chỉ số (idx) của nó
        for idx, lm in enumerate(myHand.landmark):
            # Tọa độ của landmark trong MediaPipe được chuẩn hóa trong khoảng [0, 1].
            # Cần nhân với chiều rộng và chiều cao của ảnh để có tọa độ pixel.
            cx, cy = int(lm.x * w), int(lm.y * h)
            # Thêm thông tin landmark (id, tọa độ x, tọa độ y) vào danh sách
            self.lmList.append([idx, cx, cy])
        # Trả về danh sách các vị trí landmark
        return self.lmList

    def findDistance(self, p1, p2, img=None, draw=True, r=15, t=3):
        """
        Tính toán khoảng cách Euclid giữa hai điểm landmark.

        Args:
            p1 (int): ID của điểm landmark thứ nhất.
            p2 (int): ID của điểm landmark thứ hai.
            img: Ảnh để vẽ lên (tùy chọn).
            draw (bool): Nếu True và có `img`, sẽ vẽ đường thẳng và hình tròn để minh họa khoảng cách.
            r (int): Bán kính của các hình tròn được vẽ.
            t (int): Độ dày của đường thẳng được vẽ.

        Returns:
            tuple: (length, img, info)
                - length (float or None): Khoảng cách giữa hai điểm. None nếu không thể tính.
                - img: Ảnh đã được vẽ lên (nếu có).
                - info (list): Tọa độ của hai điểm và điểm trung tâm [x1, y1, x2, y2, cx, cy].
        """
        # Đảm bảo rằng các ID landmark yêu cầu nằm trong phạm vi của lmList
        if len(self.lmList) <= max(p1, p2):
            return None, img, []
        
        # Lấy tọa độ (x, y) của hai điểm landmark
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        # Tìm tọa độ trung điểm
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Tính khoảng cách Euclid (đường chéo của tam giác vuông)
        length = math.hypot(x2 - x1, y2 - y1)

        # Nếu có ảnh và cờ `draw` là True, thực hiện vẽ để trực quan hóa
        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            
        # Trả về khoảng cách, ảnh đã vẽ, và thông tin tọa độ
        return length, img, [x1, y1, x2, y2, cx, cy]

    def fingersUp(self):
        """
        Kiểm tra xem các ngón tay đang giơ lên hay gập xuống.

        Returns:
            list: Một danh sách 5 phần tử [thumb, index, middle, ring, pinky],
                  trong đó 1 có nghĩa là ngón tay đang giơ lên, 0 là gập xuống.
        """
        # Nếu không đủ landmark để phân tích, trả về trạng thái tất cả ngón tay gập
        if len(self.lmList) < 21:
            return [0, 0, 0, 0, 0]
        
        # ID của các điểm mốc ở đầu mỗi ngón tay
        tipIds = [4, 8, 12, 16, 20]
        fingers = []

        # Xử lý ngón tay cái (thumb)
        # Logic này dựa trên tọa độ x: nếu đầu ngón tay cái (4) ở bên trái của điểm ngay dưới nó (3),
        # thì coi là ngón cái đang giơ lên. Logic này đơn giản và giả định bàn tay phải.
        # Để chính xác hơn, cần phải xác định bàn tay là trái hay phải.
        fingers.append(1 if self.lmList[tipIds[0]][1] > self.lmList[tipIds[0]-1][1] else 0)

        # Xử lý 4 ngón còn lại (trỏ, giữa, nhẫn, út)
        for i in range(1, 5):
            # So sánh tọa độ y của đầu ngón tay (tipIds[i]) và khớp ngay dưới nó (tipIds[i]-2).
            # Trong hệ tọa độ của OpenCV/Pygame, y tăng dần từ trên xuống dưới.
            # Do đó, nếu đầu ngón tay có tọa độ y nhỏ hơn, nghĩa là nó ở cao hơn, tức là đang giơ lên.
            fingers.append(1 if self.lmList[tipIds[i]][2] < self.lmList[tipIds[i]-2][2] else 0)
            
        # Trả về danh sách trạng thái của 5 ngón tay
        return fingers