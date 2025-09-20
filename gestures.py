# gestures.py
import math

def is_drawing_gesture(landmarks: list) -> bool:
    """
    Kiểm tra cử chỉ "vẽ": ngón trỏ giơ lên, các ngón khác gập lại.
    Điều này giúp cử chỉ chính xác hơn, tránh vẽ nhầm khi xòe cả bàn tay.
    """
    if not landmarks or len(landmarks) < 21:
        return False

    # Ngón trỏ giơ lên (đầu ngón cao hơn khớp giữa)
    index_finger_up = landmarks[8][2] < landmarks[6][2]
    # Ngón giữa gập xuống (đầu ngón thấp hơn khớp giữa)
    middle_finger_down = landmarks[12][2] > landmarks[10][2]
    # Ngón áp út gập xuống
    ring_finger_down = landmarks[16][2] > landmarks[14][2]

    return index_finger_up and middle_finger_down and ring_finger_down

def is_submit_gesture(landmarks: list) -> bool:
    """
    Kiểm tra cử chỉ "nộp bài": cả 5 ngón tay đều xòe ra.
    Logic được cập nhật để hoạt động với cả hai tay.
    """
    if not landmarks or len(landmarks) < 21:
        return False
    
    # --- THAY ĐỔI LOGIC NGÓN CÁI ---
    # Logic mới: Tính khoảng cách ngang giữa đầu ngón cái (4) và gốc ngón trỏ (5).
    # Khi xòe tay (cả trái và phải), khoảng cách này sẽ đủ lớn.
    # Chúng ta dùng abs() để lấy giá trị tuyệt đối, không quan tâm tay trái hay phải.
    thumb_tip_x = landmarks[4][1]
    index_finger_base_x = landmarks[5][1]
    thumb_open_distance = abs(thumb_tip_x - index_finger_base_x)
    
    # Đặt một ngưỡng, ví dụ 50 pixels. Nếu khoảng cách lớn hơn ngưỡng này, ngón cái được coi là mở.
    # Ngưỡng này có thể cần tinh chỉnh tùy thuộc vào khoảng cách của người chơi tới camera.
    thumb_up = thumb_open_distance > 50

    # Logic cho 4 ngón còn lại vẫn giữ nguyên (so sánh tọa độ Y)
    index_up = landmarks[8][2] < landmarks[6][2]
    middle_up = landmarks[12][2] < landmarks[10][2]
    ring_up = landmarks[16][2] < landmarks[14][2]
    pinky_up = landmarks[20][2] < landmarks[18][2]
    
    return all([thumb_up, index_up, middle_up, ring_up, pinky_up])

def get_cursor_position(landmarks: list):
    """Lấy vị trí của con trỏ (đầu ngón trỏ)."""
    if not landmarks or len(landmarks) < 9:
        return None
    return (landmarks[8][1], landmarks[8][2])

def is_click_gesture(landmarks: list, threshold=40):
    """
    Kiểm tra cử chỉ "click": đầu ngón trỏ và ngón giữa chụm lại.
    """
    if not landmarks or len(landmarks) < 13:
        return False
    
    # Lấy tọa độ đầu ngón trỏ (8) và ngón giữa (12)
    x1, y1 = landmarks[8][1], landmarks[8][2]
    x2, y2 = landmarks[12][1], landmarks[12][2]
    
    # Tính khoảng cách Euclid giữa chúng
    distance = math.hypot(x2 - x1, y2 - y1)
    
    return distance < threshold