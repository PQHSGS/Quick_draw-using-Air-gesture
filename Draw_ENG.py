import cv2
import mediapipe as mp
import os
import numpy as np
from keras.models import load_model
import time
from PIL import Image, ImageDraw, ImageFont

# Initialize webcam video capture
cap = cv2.VideoCapture(0)
WIDTH = 800
HEIGHT = 600
OFFSET = 20
LIMIT = 600
CENTER=(320,300)
BOX_RANGE=200
cap.set(3, WIDTH)  # Set width
cap.set(4, HEIGHT)  # Set height
cap.set(10, 150)  # Set brightness

# Initialize MediaPipe Hands object for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpdraw = mp.solutions.drawing_utils

# Initialize model
model_path = 'Quickdraw_VME\weights_ENB1V2_new.h5'  # Corrected path format for cross-platform compatibility
model = load_model(model_path)

# Load images from the folder and append them to the list
IMG_SIZE = 120
folder = 'Quickdraw_VME/color'
tools = [cv2.resize(cv2.imread(f'{folder}/{i}'), (IMG_SIZE, IMG_SIZE)) for i in os.listdir(folder)]

# Define objects
'''classes = np.array(['apple', 'banana', 'cake','cruise ship','fish','face',
       'flower', 'lantern', 'lion', 'moon', 'pear', 'pineapple', 'rabbit',
       'star', 'strawberry', 'tree', 'watermelon'])'''

classes=np.array(['Quả táo', 'Quả chuối', 'Bánh trung thu','Con tàu','Bánh cá','Mặt nạ',
       'Bông hoa', 'Đèn lồng', 'Con lân', 'Ông trăng', 'Quả lê', 'Quả dứa', 'Thỏ ngọc',
       'Đèn ông sao', 'Quả dâu tây', 'Cây thần', 'Quả dưa hấu'])
# Initialize and cache emojis with alpha masks
EMO_SIZE = 300
folder = 'Quickdraw_VME\ResNet_new'
emo = []
for i in os.listdir(folder):
    image = cv2.imread(f'{folder}/{i}', cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (EMO_SIZE, EMO_SIZE))
    if image.shape[2] == 4:  # Ensure the image has an alpha channel
        alpha_mask = image[:, :, 3]
        alpha_mask_inv = cv2.bitwise_not(alpha_mask)
        rgb_img = image[:, :, :3]
        alpha_mask = cv2.merge([alpha_mask] * 3)
        alpha_mask_inv = cv2.merge([alpha_mask_inv] * 3)
        emo.append((rgb_img, alpha_mask, alpha_mask_inv))
    else:
        emo.append((image, None, None))

#create initial drawing sequence
GAME_SIZE=30
emo_id=np.random.choice(len(classes),GAME_SIZE)
emo_list=[classes[emo] for emo in emo_id]
emo_pos = [len(emo)*15 for emo in emo_list]
#process unicode label
font_path = "arial.ttf"
font_size = 60
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
# Function to preprocess image for model prediction
def keras_process_image(canvas):
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, canvas_bw = cv2.threshold(canvas_gray, 50, 255, cv2.THRESH_BINARY)
    model_input = cv2.resize(canvas_bw, (32, 32))
    model_input = np.reshape(model_input, (1, 32, 32, 1))
    return model_input

# Function to predict the drawn image
def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed, verbose=0)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

# Function to check if all fingers are open
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
    eraser, pen = tools[0], tools[1]
    xp, yp = 0, 0
    is_saved = False
    is_draw = False
    is_spam = False
    col = (0, 255, 255)  # Default color (yellow)
    brush_size=25
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    start_point = (CENTER[0]-BOX_RANGE, CENTER[1]-BOX_RANGE)
    end_point = (CENTER[0]+BOX_RANGE, CENTER[1]+BOX_RANGE)    
    result_icon = None
    display_time = 0
    frame_count = 0
    target=''
    target_pos=0
    target_id=0
    count=-1
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=5)
        cv2.rectangle(frame,(LIMIT+OFFSET,HEIGHT//2-OFFSET*3-IMG_SIZE),(LIMIT+OFFSET+IMG_SIZE,HEIGHT//2-OFFSET*3),color=(0,0,0),thickness=5)
        cv2.rectangle(frame,(LIMIT+OFFSET,HEIGHT//2+OFFSET*3),(LIMIT+OFFSET+IMG_SIZE,HEIGHT//2+OFFSET*3+IMG_SIZE),color=(0,0,0),thickness=5)
        if is_spam:
            count+=1
            if count < len(emo_list):  # Ensure count is within bounds
                target = emo_list[count]
                target_pos = emo_pos[count]
                target_id=emo_id[count]
                is_spam=False
            else:
                print("Reached the end of emo_list.")
                is_spam = False
        #cv2.putText(frame,target,(CENTER[0]-target_pos,(CENTER[1]-BOX_RANGE)//3*2),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=2,color=(255,0,0),thickness=2)
        frame = putTextUnicode(frame, target, (CENTER[0]-target_pos,(CENTER[1]-BOX_RANGE)//5), font_path=font_path, font_size=font_size, color=(0, 255, 255))
        # Process every 2nd frame to reduce load
        if frame_count % 2 == 0:           
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

                if check_fingers_open(lanmark) and is_draw and not is_saved:
                    box = canvas[CENTER[1]-BOX_RANGE:CENTER[1]+BOX_RANGE, CENTER[0]-BOX_RANGE:CENTER[0]+BOX_RANGE]
                    prob, class_label = keras_predict(model, box)
                    result_icon = emo[class_label]
                    display_time = time.time()
                    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                    is_saved = True
                    is_draw=False
                    if class_label==target_id:
                        is_spam=True
                elif lanmark[8][2] < lanmark[6][2] and lanmark[12][2] < lanmark[10][2]:
                    is_saved = False
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

                elif lanmark[8][2] < lanmark[6][2]:
                    is_draw=True
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    #cv2.line(frame, (xp, yp), (x1, y1), col, brush_size, cv2.FILLED)
                    cv2.line(canvas, (xp, yp), (x1, y1), col, brush_size, cv2.FILLED)
                    xp, yp = x1, y1

        # Update eraser and pen positions
        eraser_pos = (LIMIT + OFFSET, HEIGHT // 2 - OFFSET * 3)
        pen_pos = (LIMIT + OFFSET, HEIGHT // 2 + OFFSET * 3)
        frame[eraser_pos[1] - IMG_SIZE:eraser_pos[1], eraser_pos[0]:eraser_pos[0] + IMG_SIZE] = eraser
        frame[pen_pos[1]:pen_pos[1] + IMG_SIZE, pen_pos[0]:pen_pos[0] + IMG_SIZE] = pen

        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

        if imgInv.shape != frame.shape:
            imgInv = cv2.resize(imgInv, (frame.shape[1], frame.shape[0]))

        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)

        # Display the icon for 2 seconds
        if result_icon is not None and time.time() - display_time < 2:
            overlay_icon(frame, result_icon)

        cv2.imshow('cam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('p'):
            is_spam=True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
