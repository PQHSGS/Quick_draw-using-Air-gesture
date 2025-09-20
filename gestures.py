# gestures.py
# Module chứa các hàm nhận diện cử chỉ từ landmarks của bàn tay

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
    """
    if not landmarks or len(landmarks) < 21:
        return False
    
    # Logic giống hệt file Draw_VIE.py
    # Lưu ý: logic cho ngón cái có thể cần điều chỉnh tùy theo góc camera
    thumb_up = landmarks[4][1] < landmarks[3][1] # Cho bàn tay phải
    index_up = landmarks[8][2] < landmarks[6][2]
    middle_up = landmarks[12][2] < landmarks[10][2]
    ring_up = landmarks[16][2] < landmarks[14][2]
    pinky_up = landmarks[20][2] < landmarks[18][2]
    
    return all([thumb_up, index_up, middle_up, ring_up, pinky_up])