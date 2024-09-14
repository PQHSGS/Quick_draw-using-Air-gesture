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
import asyncio
from concurrent.futures import ThreadPoolExecutor
from cloudinary.uploader import upload
from datetime import datetime

# Hàm để upload ảnh
def upload_image(path, name):
    return upload(path, public_id=f"{datetime.now().strftime('%w_%H_%M_%S')}_{name}")

# Hàm bất đồng bộ cho upload ảnh
async def async_upload_image(executor, path, name):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, upload_image, path, name)
    return result

cloudinary.config( 
    cloud_name = "dooyfe3ar", 
    api_key = "511834484476363", 
    api_secret = "arMWPkXJdnn2xtSEPcDog2zfg34",
    secure=True
)

# CÁC THÔNG SỐ CƠ BẢN
cap = cv2.VideoCapture(0)
WIDTH = 800
HEIGHT = 600
OFFSET = 20
LIMIT = 600
CENTER = (400, 250)
BOX_RANGE = 125
cap.set(3, WIDTH)
cap.set(4, HEIGHT)
cap.set(10, 150)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpdraw = mp.solutions.drawing_utils

model_path = 'weights_VGG_new.h5'
model = load_model(model_path)

IMG_SIZE = 120
folder = 'tools' 
tools = [cv2.resize(cv2.imread(f'{folder}/{i}'), (IMG_SIZE, IMG_SIZE)) for i in os.listdir(folder)]
logo = cv2.resize(cv2.imread('vme.jpg'), (70, 70))

classes_eng = np.array(['apple', 'banana', 'cake', 'cruise_ship', 'fish', 'face',
                        'flower', 'lantern', 'lion', 'moon', 'pear', 'pineapple', 'rabbit',
                        'star', 'strawberry', 'tree', 'watermelon'])
classes = np.array(['Quả táo', 'Quả chuối', 'Bánh trung thu', 'Con tàu', 'Mặt nạ', 'Bánh cá',
                    'Bông hoa', 'Đèn lồng', 'Con lân', 'Ông trăng', 'Quả lê', 'Quả dứa', 'Thỏ ngọc',
                    'Đèn ông sao', 'Quả dâu tây', 'Cây thần', 'Quả dưa hấu'])
classes_len = np.array([len(i)*12 for i in classes])

EMO_SIZE = BOX_RANGE*3//2
folder = "icon_v2"
emo = []
for i in os.listdir(folder):
    for j in os.listdir(f"{folder}/{i}"):
        image = cv2.imread(f"{folder}/{i}/{j}", cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (EMO_SIZE, EMO_SIZE))
        if image.shape[2] == 4:
            alpha_mask = image[:, :, 3]
            alpha_mask_inv = cv2.bitwise_not(alpha_mask)
            rgb_img = image[:, :, :3]
            alpha_mask = cv2.merge([alpha_mask] * 3)
            alpha_mask_inv = cv2.merge([alpha_mask_inv] * 3)
            emo.append((rgb_img, alpha_mask, alpha_mask_inv))
        else:
            emo.append((image, None, None))
# for i in os.listdir(folder):
#     image = cv2.imread(f'{folder}/{i}', cv2.IMREAD_UNCHANGED)
#     image = cv2.resize(image, (EMO_SIZE, EMO_SIZE))
#     if image.shape[2] == 4:
#         alpha_mask = image[:, :, 3]
#         alpha_mask_inv = cv2.bitwise_not(alpha_mask)
#         rgb_img = image[:, :, :3]
#         alpha_mask = cv2.merge([alpha_mask] * 3)
#         alpha_mask_inv = cv2.merge([alpha_mask_inv] * 3)
#         emo.append((rgb_img, alpha_mask, alpha_mask_inv))
#     else:
#         emo.append((image, None, None))

def init_game():
    global name, word_len, eraser, pen, xp, yp, is_saved, is_draw, col, brush_size, canvas, start_point, end_point, result_icon, display_time, frame_count
    eraser, pen = tools[0], tools[1]
    name = ""
    word_len = len(name)*12
    xp, yp = 0, 0
    is_saved = False
    is_draw = False
    col = (0, 255, 255)
    brush_size = 10
    canvas = np.zeros((480, 848, 3), np.uint8)
    start_point = (CENTER[0] - BOX_RANGE, CENTER[1] - BOX_RANGE)
    end_point = (CENTER[0] + BOX_RANGE, CENTER[1] + BOX_RANGE)    
    result_icon = None
    display_time = 0
    frame_count = 0

font_path = "arial.ttf"
font_size = 45
def putTextUnicode(img, text, position, font_path='arial.ttf', font_size=30, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def keras_process_image(canvas):
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, canvas_bw = cv2.threshold(canvas_gray, 50, 255, cv2.THRESH_BINARY)
    model_input = cv2.resize(canvas_bw, (32, 32))
    model_input = np.reshape(model_input, (1, 32, 32, 1))
    return model_input

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred = np.argmax(model.predict(processed, verbose=0)[0])
    prob = np.max(model.predict(processed, verbose=0)[0])
    return pred, prob

def check_fingers_open(lanmark):
    if len(lanmark) < 21:
        return False
    conditions = [
        lanmark[4][1] < lanmark[3][1],
        lanmark[8][2] < lanmark[6][2],
        lanmark[12][2] < lanmark[10][2],
        lanmark[16][2] < lanmark[14][2],
        lanmark[20][2] < lanmark[18][2]
    ]
    return all(conditions)

def overlay_icon(frame, icon_info, pos=(CENTER[0] - EMO_SIZE//2, CENTER[1] - EMO_SIZE//2)):
    rgb_img, alpha_mask, alpha_mask_inv = icon_info
    icon_h, icon_w = rgb_img.shape[:2]
    roi = frame[pos[1]:pos[1] + icon_h, pos[0]:pos[0] + icon_w]

    if alpha_mask is not None:
        roi_bg = cv2.bitwise_and(roi, roi, mask=alpha_mask_inv[:, :, 0])
        icon_fg = cv2.bitwise_and(rgb_img, rgb_img, mask=alpha_mask[:, :, 0])
        dst = cv2.add(roi_bg, icon_fg)
    else:
        dst = cv2.addWeighted(roi, 0.1, rgb_img, 0.9, 0)
    frame[pos[1]:pos[1] + icon_h, pos[0]:pos[0] + icon_w] = dst

async def main_loop():
    global name, word_len, eraser, pen, xp, yp, is_saved, is_draw, col, brush_size, canvas, start_point, end_point, result_icon, display_time, frame_count
    NUM_PER_CLASS = 2
    OVER=125
    init_game()
    
    cv2.namedWindow('cam', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('cam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    executor = ThreadPoolExecutor(max_workers=5)

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        #print(frame.shape[0], frame.shape[1])
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=5)
        # cv2.rectangle(frame, (LIMIT + OFFSET, HEIGHT // 2 - OFFSET * 3 - IMG_SIZE), 
        #               (LIMIT + OFFSET + IMG_SIZE, HEIGHT // 2 - OFFSET * 3), color=(0, 0, 0), thickness=5)
        # cv2.rectangle(frame, (LIMIT + OFFSET, HEIGHT // 2 + OFFSET * 3), 
        #               (LIMIT + OFFSET + IMG_SIZE, HEIGHT // 2 + OFFSET * 3 + IMG_SIZE), color=(0, 0, 0), thickness=5)
        
        if frame_count % 2 == 0:
            region = img[CENTER[1] - BOX_RANGE - OVER:CENTER[1] + BOX_RANGE + OVER, 
                         CENTER[0] - BOX_RANGE - OVER:CENTER[0] + BOX_RANGE + OVER]
            results = hands.process(region)
            lanmark = []
            if results.multi_hand_landmarks:
                for hn in results.multi_hand_landmarks:
                    for id, lm in enumerate(hn.landmark):
                        h, w, c = region.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lanmark.append([id, cx, cy])

            if len(lanmark) != 0:
                x1, y1 = lanmark[8][1], lanmark[8][2]
                x2, y2 = lanmark[12][1], lanmark[12][2]

                # Align coordinates with the canvas region
                x1 += CENTER[0] - BOX_RANGE-OVER
                y1 += CENTER[1] - BOX_RANGE-OVER
                x2 += CENTER[0] - BOX_RANGE-OVER
                y2 += CENTER[1] - BOX_RANGE-OVER

                if check_fingers_open(lanmark) and is_draw and not is_saved:
                    box = canvas[CENTER[1] - BOX_RANGE:CENTER[1] + BOX_RANGE, CENTER[0] - BOX_RANGE:CENTER[0] + BOX_RANGE]
                    class_label, prob = keras_predict(model, box)
                    display_time = time.time()
                    canvas = np.zeros((480, 848, 3), np.uint8)
                    is_saved = True
                    is_draw = False
                    print(f"Predicted: {classes[class_label]} with prob: {prob}")
                    if prob < 0.4:
                        result_icon = None
                        name = "Không nhận diện được"
                        word_len = len(name) * 12
                    else:
                        #result_icon = emo[class_label]
                        num=np.random.randint(1, NUM_PER_CLASS+1)
                        name = classes[class_label]
                        word_len = classes_len[class_label]
                        result_icon = emo[(class_label)*NUM_PER_CLASS+num-1]
                        path = f"icon_v2\\{classes_eng[class_label]}\\{num}.png"
                        #result_icon = cv2.resize(cv2.imread(path, cv2.IMREAD_UNCHANGED), (EMO_SIZE, EMO_SIZE))
                        await async_upload_image(executor, path, name)

                elif lanmark[8][2] < lanmark[6][2] and lanmark[12][2] < lanmark[10][2]:
                    xp, yp = 0, 0
                    # if x1 > LIMIT:
                    #     if HEIGHT / 2 - OFFSET * 3 > y1 > HEIGHT / 2 - (OFFSET * 3 + IMG_SIZE):
                    #         col = (0, 0, 0)
                    #         brush_size = 50
                    #         cv2.rectangle(frame, (LIMIT + OFFSET, HEIGHT // 2 - OFFSET * 3 - IMG_SIZE), 
                    #                       (LIMIT + OFFSET + IMG_SIZE, HEIGHT // 2 - OFFSET * 3), color=(0, 255, 0), thickness=5)
                    #     if HEIGHT / 2 + OFFSET * 3 < y1 < HEIGHT / 2 + OFFSET * 3 + IMG_SIZE:
                    #         col = (0, 255, 255)
                    #         brush_size = 25
                    #         cv2.rectangle(frame, (LIMIT + OFFSET, HEIGHT // 2 + OFFSET * 3), 
                    #                       (LIMIT + OFFSET + IMG_SIZE, HEIGHT // 2 + OFFSET * 3 + IMG_SIZE), color=(0, 255, 0), thickness=5)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, cv2.FILLED)

                elif lanmark[8][2] < lanmark[6][2]:
                    is_saved = False
                    is_draw = True
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    cv2.line(canvas, (xp, yp), (x1, y1), col, brush_size, cv2.FILLED)
                    xp, yp = x1, y1

        # eraser_pos = (LIMIT + OFFSET, HEIGHT // 2 - OFFSET * 3)
        # pen_pos = (LIMIT + OFFSET, HEIGHT // 2 + OFFSET * 3)
        # frame[eraser_pos[1] - IMG_SIZE:eraser_pos[1], eraser_pos[0]:eraser_pos[0] + IMG_SIZE] = eraser
        # frame[pen_pos[1]:pen_pos[1] + IMG_SIZE, pen_pos[0]:pen_pos[0] + IMG_SIZE] = pen
        frame[385:455,25:95]=logo
        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        imgInv = cv2.resize(imgInv, (frame.shape[1], frame.shape[0])) if imgInv.shape != frame.shape else imgInv
        frame = cv2.bitwise_and(frame, imgInv)

        if result_icon is not None and time.time() - display_time < 4:
            overlay_icon(frame, result_icon)
        frame = putTextUnicode(frame, name, (CENTER[0] - word_len, (CENTER[1] - BOX_RANGE) // 4), font_path=font_path, font_size=font_size, color=(0, 0, 255))

        cv2.imshow('cam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

asyncio.run(main_loop())
