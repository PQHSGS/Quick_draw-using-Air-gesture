# HandTrack.py

# Import các thư viện cần thiết
import math # Thư viện toán học để tính toán, ví dụ như khoảng cách
import cv2 # Thư viện OpenCV để xử lý hình ảnh và video
import mediapipe as mp # Thư viện của Google để nhận dạng bàn tay và các điểm mốc
import time # Thư viện thời gian để tính toán FPS (khung hình trên giây)

# Tạo một lớp (class) để nhận diện bàn tay
class handDetector():
    # Hàm khởi tạo (constructor) của lớp, được gọi khi một đối tượng mới được tạo
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        # --- CÁC THAM SỐ KHỞI TẠO CHO BỘ NHẬN DIỆN BÀN TAY ---
        self.mode = mode # Chế độ tĩnh (static mode): True (chậm hơn) hoặc False (nhanh hơn, theo dõi liên tục)
        self.maxHands = maxHands # Số lượng bàn tay tối đa có thể nhận diện
        self.modelComplex = modelComplexity # Độ phức tạp của mô hình nhận diện (0 hoặc 1)
        self.detectionCon = detectionCon # Ngưỡng tin cậy tối thiểu để nhận diện một bàn tay (0.0 -> 1.0)
        self.trackCon = trackCon # Ngưỡng tin cậy tối thiểu để theo dõi vị trí bàn tay (0.0 -> 1.0)

        # --- KHỞI TẠO CÁC ĐỐI TƯỢNG TỪ MEDIAPIPE ---
        # Khởi tạo giải pháp nhận diện bàn tay của Mediapipe
        self.mpHands = mp.solutions.hands
        # Tạo đối tượng Hands với các tham số đã được cấu hình ở trên
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        # Khởi tạo công cụ để vẽ các điểm mốc và đường nối trên bàn tay
        self.mpDraw = mp.solutions.drawing_utils
        # Danh sách ID của các đầu ngón tay (ngón cái, trỏ, giữa, áp út, út)
        self.tipIds = [4, 8, 12, 16, 20]

    # Hàm tìm và vẽ các bàn tay trong một khung hình (image)
    def findHands(self, img, draw=True):
        # Chuyển đổi không gian màu của ảnh từ BGR (OpenCV) sang RGB (Mediapipe)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Xử lý ảnh RGB để tìm kiếm bàn tay, kết quả được lưu vào self.results
        self.results = self.hands.process(imgRGB)
        
        # Kiểm tra xem có tìm thấy bàn tay nào không
        if self.results.multi_hand_landmarks:
            # Lặp qua từng bàn tay được phát hiện
            for handLms in self.results.multi_hand_landmarks:
                # Nếu cờ 'draw' là True, vẽ các điểm mốc (landmarks) và đường nối
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        # Trả về khung hình đã được vẽ (hoặc không)
        return img

    # Hàm tính khoảng cách giữa hai điểm mốc (landmark) trên bàn tay
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        # Lấy tọa độ x, y của điểm thứ nhất (p1) từ danh sách các điểm mốc
        x1, y1 = self.lmList[p1][1:]
        # Lấy tọa độ x, y của điểm thứ hai (p2)
        x2, y2 = self.lmList[p2][1:]
        # Tìm tọa độ trung điểm của đường thẳng nối hai điểm
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Nếu cờ 'draw' là True, vẽ các chi tiết lên ảnh
        if draw:
            # Vẽ đường thẳng nối hai điểm
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            # Vẽ vòng tròn tại điểm thứ nhất
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            # Vẽ vòng tròn tại điểm thứ hai
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            # Vẽ vòng tròn tại trung điểm
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        
        # Tính toán khoảng cách Euclid giữa hai điểm
        length = math.hypot(x2 - x1, y2 - y1)

        # Trả về khoảng cách, ảnh đã vẽ và danh sách các tọa độ liên quan
        return length, img, [x1, y1, x2, y2, cx, cy]

    # Hàm tìm và trả về vị trí của tất cả các điểm mốc trên một bàn tay cụ thể
    def findPosition(self, img, handNo=0, draw=True):
        # Khởi tạo một danh sách rỗng để lưu trữ vị trí các điểm mốc
        self.lmlist = []
        # Kiểm tra xem có tìm thấy bàn tay nào không
        if self.results.multi_hand_landmarks:
            # Chọn bàn tay cần xử lý (mặc định là bàn tay đầu tiên, handNo=0)
            myHand = self.results.multi_hand_landmarks[handNo]
            # Lặp qua từng điểm mốc (landmark) trên bàn tay đã chọn
            for id, lm in enumerate(myHand.landmark):
                # Lấy chiều cao (h), chiều rộng (w) của ảnh
                h, w, c = img.shape
                # Chuyển đổi tọa độ tương đối (0.0 -> 1.0) của điểm mốc thành tọa độ pixel
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Thêm ID và tọa độ pixel của điểm mốc vào danh sách
                self.lmlist.append([id, cx, cy])
        # Trả về danh sách chứa vị trí của tất cả các điểm mốc
        return self.lmlist

    # Hàm kiểm tra xem các ngón tay đang giơ lên hay gập xuống
    def fingersUp(self):
        fingers = [] # Danh sách để lưu trạng thái của 5 ngón tay (1: giơ, 0: gập)
        
        # --- KIỂM TRA NGÓN CÁI ---
        # So sánh tọa độ x của đầu ngón cái (tipIds[0]) và điểm ngay trước đó
        # Logic này đúng cho bàn tay phải khi nhìn từ trước
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1) # Ngón cái giơ
        else:
            fingers.append(0) # Ngón cái gập
            
        # --- KIỂM TRA 4 NGÓN CÒN LẠI ---
        # Lặp qua các ngón trỏ, giữa, áp út, út
        for id in range(1, 5):
            # So sánh tọa độ y của đầu ngón tay với điểm mốc ở khớp giữa của ngón đó
            if (self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]):
                fingers.append(1) # Ngón tay giơ
            else:
                fingers.append(0) # Ngón tay gập
                
        # Trả về danh sách trạng thái của 5 ngón tay
        return fingers

# Hàm chính để chạy chương trình
def main():
    # pTime: thời gian của khung hình trước đó (previous time)
    pTime = 0
    # cTime: thời gian của khung hình hiện tại (current time)
    cTime = 0
    # Mở camera mặc định (số 0) của máy tính
    cap = cv2.VideoCapture(0)
    # Tạo một đối tượng từ lớp handDetector
    detector = handDetector()
    
    # Bắt đầu vòng lặp vô hạn để đọc video từ camera
    while True:
        # Đọc một khung hình từ camera
        success, img = cap.read()
        # Nếu không đọc được khung hình, thoát khỏi vòng lặp
        if not success:
            break
        
        # Sử dụng đối tượng detector để tìm và vẽ bàn tay lên khung hình
        img = detector.findHands(img)
        # Tìm vị trí các điểm mốc trên bàn tay
        lmlist = detector.findPosition(img)
        
        # --- TÍNH TOÁN VÀ HIỂN THỊ FPS ---
        # Lấy thời gian hiện tại
        cTime = time.time()
        # Tính FPS = 1 / (thời gian xử lý một khung hình)
        fps = 1 / (cTime - pTime)
        # Cập nhật pTime bằng cTime cho lần lặp tiếp theo
        pTime = cTime
        # Vẽ giá trị FPS lên góc trên bên trái của khung hình
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        # Hiển thị khung hình đã xử lý trong một cửa sổ có tên "Image"
        cv2.imshow("Image", img)
        # Chờ 1 mili giây; nếu có phím nào được nhấn, vòng lặp sẽ dừng
        cv2.waitKey(1)

# Kiểm tra xem file này có đang được chạy trực tiếp hay không
if __name__ == "__main__":
    main() # Nếu có, gọi hàm main() để bắt đầu chương trình

# Giải phóng tài nguyên và đóng tất cả các cửa sổ của OpenCV sau khi chương trình kết thúc
cv2.destroyAllWindows()