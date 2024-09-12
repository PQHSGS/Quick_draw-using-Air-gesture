import cv2
import mediapipe as mp
import os
import numpy as np
from keras.models import load_model
import time
from datetime import datetime
import cloudinary.uploader
import cloudinary.api
from PIL import Image, ImageDraw, ImageFont

cloudinary.config( 
    cloud_name = "dwsk1vwlc", 
    api_key = "234436162877572", 
    api_secret = "PwGo4ddZ2GnEdjm2Yd1Y2NW7OfA", # Click 'View API Keys' above to copy your API secret
    secure=True
)
#CÁC THÔNG SỐ CƠ BẢN

# Khởi tạo thông số cam
cap = cv2.VideoCapture(0)
WIDTH = 800
HEIGHT = 600
OFFSET = 20
LIMIT = 600
CENTER=(340,300) #Tọa độ ô vẽ
BOX_RANGE=225 #Chiều rộng ô
cap.set(3, WIDTH) 
cap.set(4, HEIGHT)
cap.set(10, 150)  #độ sáng

# Mediapipe - phát hiện tay
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpdraw = mp.solutions.drawing_utils

# Mô hình dự đoán
model_path = 'weights_VGG_new.h5'
model = load_model(model_path)

# Ảnh công cụ bút & tẩy
IMG_SIZE = 120
folder = 'tools' 
tools = [cv2.resize(cv2.imread(f'{folder}/{i}'), (IMG_SIZE, IMG_SIZE)) for i in os.listdir(folder)]
logo = cv2.resize(cv2.imread('vme.jpg'),(70,70))
# Các đồ vật có thể nhận diện được
classes_eng=np.array(['apple', 'banana', 'cake','cruise_ship','fish','face',
       'flower', 'lantern', 'lion', 'moon', 'pear', 'pineapple', 'rabbit',
       'star', 'strawberry', 'tree', 'watermelon'])
# Tên bằng tiếng việt
classes=np.array(['Quả táo', 'Quả chuối', 'Bánh trung thu','Con tàu','Mặt nạ', 'Bánh cá',
       'Bông hoa', 'Đèn lồng', 'Con lân', 'Ông trăng', 'Quả lê', 'Quả dứa', 'Thỏ ngọc',
       'Đèn ông sao', 'Quả dâu tây', 'Cây thần', 'Quả dưa hấu'])
classes_len = np.array([len(i)*12 for i in classes])
# Load các hình đồ vật và xử lí để chèn lên màn hình
EMO_SIZE = 300
folder = "icon_v1"
emo = []
for i in os.listdir(folder):
    image = cv2.imread(f'{folder}/{i}', cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (EMO_SIZE, EMO_SIZE))
    if image.shape[2] == 4:  # check alpha channel của ảnh
        alpha_mask = image[:, :, 3]
        alpha_mask_inv = cv2.bitwise_not(alpha_mask)
        rgb_img = image[:, :, :3]
        alpha_mask = cv2.merge([alpha_mask] * 3)
        alpha_mask_inv = cv2.merge([alpha_mask_inv] * 3)
        emo.append((rgb_img, alpha_mask, alpha_mask_inv))
    else:
        emo.append((image, None, None))


#CÁC HÀM CẦN THIẾT
#Khởi tạo các thông số trước để tránh lag
# def upload_image_to_cloudinary(image, image_name="predicted_image"):
#     """Uploads the image to Cloudinary."""
#     _, img_encoded = cv2.imencode('.png', image)  # Save as PNG for transparency
#     img_bytes = img_encoded.tobytes()

#     response = cloudinary.uploader.upload(img_bytes, public_id=image_name, resource_type="image", format="png")
#     image_url = response['secure_url']
#     return image_url

def init_game():
    global name, word_len, eraser, pen, xp, yp, is_saved, is_draw, col, brush_size, canvas, start_point, end_point, result_icon, display_time, frame_count
    eraser, pen = tools[0], tools[1]
    name = ""
    word_len = len(name)*12
    xp, yp = 0, 0 # tọa độ ngón tay trước đó để vẽ thành các nét line siêu ngắn (tăng độ mượt)
    is_saved = False #flag để tránh check tay mở liên tục
    is_draw = False #flag xác nhận người chơi đã vẽ j đó trc khi nộp
    col = (0, 255, 255)  # Default color (yellow)
    brush_size=25 #độ lớn cọ vẽ ban đầu
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8) #ảnh sẽ được vẽ lên đây
    start_point = (CENTER[0]-BOX_RANGE, CENTER[1]-BOX_RANGE)
    end_point = (CENTER[0]+BOX_RANGE, CENTER[1]+BOX_RANGE)    
    result_icon = None #hình vẽ dự đoán được mỗi lần nộp
    display_time = 0 #flag để cố định thời gian icon hiện lên
    frame_count = 0
#xử lí tập class bên trên để cv2 có thể đẩy dc chữ unicode lên màn hình
font_path = "arial.ttf"
font_size = 45
def putTextUnicode(img, text, position, font_path='arial.ttf', font_size=30, color=(255, 255, 255)):
    """
    Draw Unicode text on an image using PIL and convert it to OpenCV format.

    :param img: OpenCV image
    :param text: Text to draw (Unicode supported)
    :param position: Position tuple (x, y) to place the text
    :param font_path: Path to a .ttf font file that supports Unicode characters
    :param font_size: Size of the font
    :param color: Text color in BGR format
    :return: Image with the drawn text
    """
    # Convert OpenCV image (numpy array) to PIL image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Create a drawing context
    draw = ImageDraw.Draw(img_pil)
    # Load the desired font
    font = ImageFont.truetype(font_path, font_size)
    # Draw the text
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))  # Convert BGR to RGB
    # Convert back to OpenCV image
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

# Tiền xử lý ảnh vẽ
def keras_process_image(canvas):
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, canvas_bw = cv2.threshold(canvas_gray, 50, 255, cv2.THRESH_BINARY)
    model_input = cv2.resize(canvas_bw, (32, 32))
    model_input = np.reshape(model_input, (1, 32, 32, 1))
    return model_input

# Dự đoán hình vẽ
def keras_predict(model, image):
    processed = keras_process_image(image)
    # Get predicted class
    pred = np.argmax(model.predict(processed, verbose=0)[0])
    # Get probability of predicted class
    prob = np.max(model.predict(processed, verbose=0)[0])
    return pred, prob

# Kiểm tra dấu hiệu nộp ảnh - mở tay
def check_fingers_open(lanmark):
    if len(lanmark) < 21:
        return False
    conditions = [
        lanmark[4][1] < lanmark[3][1],    # Thumb
        lanmark[8][2] < lanmark[6][2],    # Index finger
        lanmark[12][2] < lanmark[10][2],  # Middle finger
        lanmark[16][2] < lanmark[14][2],  # Ring finger
        lanmark[20][2] < lanmark[18][2]   # Pinky
    ]
    return all(conditions)
#Overlay ảnh icon đồ vật
def overlay_icon(frame, icon_info, pos=(CENTER[0]-EMO_SIZE//2, CENTER[1]-EMO_SIZE//2)):
    rgb_img, alpha_mask, alpha_mask_inv = icon_info
    icon_h, icon_w = rgb_img.shape[:2]
    roi = frame[pos[1]:pos[1] + icon_h, pos[0]:pos[0] + icon_w]

    if alpha_mask is not None:
        roi_bg = cv2.bitwise_and(roi, roi, mask=alpha_mask_inv[:, :, 0])
        icon_fg = cv2.bitwise_and(rgb_img, rgb_img, mask=alpha_mask[:, :, 0])
        dst = cv2.add(roi_bg, icon_fg)
    else:
        dst = cv2.addWeighted(roi, 1, rgb_img, 0.7, 0)
    frame[pos[1]:pos[1] + icon_h, pos[0]:pos[0] + icon_w] = dst
    
def main():
    global name, word_len, eraser, pen, xp, yp, is_saved, is_draw, col, brush_size, canvas, start_point, end_point, result_icon, display_time, frame_count
    init_game()
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image. Exiting.")
            break
        #Xử lý ảnh
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=5)
        cv2.rectangle(frame,(LIMIT+OFFSET,HEIGHT//2-OFFSET*3-IMG_SIZE),(LIMIT+OFFSET+IMG_SIZE,HEIGHT//2-OFFSET*3),color=(0,0,0),thickness=5)
        cv2.rectangle(frame,(LIMIT+OFFSET,HEIGHT//2+OFFSET*3),(LIMIT+OFFSET+IMG_SIZE,HEIGHT//2+OFFSET*3+IMG_SIZE),color=(0,0,0),thickness=5)
        
        #Bắt đầu game
        # Chỉ xử lý các frame chẵn
        if frame_count % 2 == 0:  
            #Xác định vị trí tay         
            results = hands.process(img)
            lanmark = []
            if results.multi_hand_landmarks:
                for hn in results.multi_hand_landmarks:
                    for id, lm in enumerate(hn.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lanmark.append([id, cx, cy])
                    #mpdraw.draw_landmarks(frame, hn, mpHands.HAND_CONNECTIONS)

            if len(lanmark) != 0:
                x1, y1 = lanmark[8][1], lanmark[8][2]
                x2, y2 = lanmark[12][1], lanmark[12][2]
                #tiêu chí nộp bài
                if check_fingers_open(lanmark) and is_draw and not is_saved:
                    box = canvas[CENTER[1]-BOX_RANGE:CENTER[1]+BOX_RANGE, CENTER[0]-BOX_RANGE:CENTER[0]+BOX_RANGE]
                    class_label, prob = keras_predict(model, box)
                    display_time = time.time()
                    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                    is_saved = True
                    is_draw=False
                    print(f"Predicted: {classes[class_label]} with prob: {prob}")
                    if prob <0.4:
                        result_icon = None
                        name="Không nhận diện được"
                        word_len = len(name)*12
                    else:
                        result_icon = emo[class_label]
                        name=classes[class_label]  
                        word_len = classes_len[class_label]  
                        path=f"icon_v1\{classes_eng[class_label]}.png"
                        cloudinary.uploader.upload(path, public_id=f"{datetime.now().strftime('%w_%H_%M_%S')}_{name}")
                # tiêu chí thay đổi công cụ
                elif lanmark[8][2] < lanmark[6][2] and lanmark[12][2] < lanmark[10][2]:
                    xp, yp = 0, 0
                    if x1 > LIMIT:
                        if HEIGHT / 2 - OFFSET*3 > y1 > HEIGHT / 2 - (OFFSET*3 + IMG_SIZE):
                            col = (0, 0, 0)  # Eraser
                            brush_size=50
                            cv2.rectangle(frame,(LIMIT+OFFSET,HEIGHT//2-OFFSET*3-IMG_SIZE),(LIMIT+OFFSET+IMG_SIZE,HEIGHT//2-OFFSET*3),color=(0,255,0),thickness=5)
                        if HEIGHT / 2 + OFFSET*3 < y1 < HEIGHT / 2 + OFFSET*3 + IMG_SIZE:
                            col = (0, 255, 255)  # Pen (yellow)
                            brush_size=25
                            cv2.rectangle(frame,(LIMIT+OFFSET,HEIGHT//2+OFFSET*3),(LIMIT+OFFSET+IMG_SIZE,HEIGHT//2+OFFSET*3+IMG_SIZE),color=(0,255,0),thickness=5)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, cv2.FILLED)
                #tiêu chí vẽ
                elif lanmark[8][2] < lanmark[6][2]:
                    is_saved = False
                    is_draw=True
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    #cv2.line(frame, (xp, yp), (x1, y1), col, brush_size, cv2.FILLED)
                    cv2.line(canvas, (xp, yp), (x1, y1), col, brush_size, cv2.FILLED)
                    xp, yp = x1, y1

        # Khởi tạo vị trí thanh công cụ
        eraser_pos = (LIMIT + OFFSET, HEIGHT // 2 - OFFSET * 3)
        pen_pos = (LIMIT + OFFSET, HEIGHT // 2 + OFFSET * 3)
        frame[eraser_pos[1] - IMG_SIZE:eraser_pos[1], eraser_pos[0]:eraser_pos[0] + IMG_SIZE] = eraser
        frame[pen_pos[1]:pen_pos[1] + IMG_SIZE, pen_pos[0]:pen_pos[0] + IMG_SIZE] = pen
        frame[525:595,10:80]=logo
        #blend canvas và frame với nhau để hiện hình vẽ lên cam quay
        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

        if imgInv.shape != frame.shape:
            imgInv = cv2.resize(imgInv, (frame.shape[1], frame.shape[0]))

        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)

        # Hiện icon trong vòng 2 giây
        if result_icon is not None and time.time() - display_time < 2:
            overlay_icon(frame, result_icon)
        frame = putTextUnicode(frame, name, (CENTER[0]-word_len,(CENTER[1]-BOX_RANGE)//5), font_path=font_path, font_size=font_size, color=(0, 0, 255))
        
        cv2.imshow('cam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #Hủy game
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
