import cv2
import mediapipe as mp
import os
import numpy as np
from keras.models import load_model
import time
import zipfile
import cloudinary.uploader
import cloudinary.api
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
#CÁC THÔNG SỐ CƠ BẢN
cloudinary.config( 
    cloud_name = "dwsk1vwlc", 
    api_key = "234436162877572", 
    api_secret = "PwGo4ddZ2GnEdjm2Yd1Y2NW7OfA", # Click 'View API Keys' above to copy your API secret
    secure=True
)
# Khởi tạo thông số cam
cap = cv2.VideoCapture(0)
WIDTH = 800
HEIGHT = 600
OFFSET = 20
LIMIT = 600
CENTER=(340,300) #Tọa độ ô vẽ
BOX_RANGE=225 #Chiều rộng ô

player_id=1
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
'''classes = np.array(['apple', 'banana', 'cake','cruise ship','fish','face',
       'flower', 'lantern', 'lion', 'moon', 'pear', 'pineapple', 'rabbit',
       'star', 'strawberry', 'tree', 'watermelon'])'''
# Tên bằng tiếng việt
classes=np.array(['Quả táo', 'Quả chuối', 'Bánh trung thu','Con tàu','Bánh cá','Mặt nạ',
       'Bông hoa', 'Đèn lồng', 'Con lân', 'Ông trăng', 'Quả lê', 'Quả dứa', 'Thỏ ngọc',
       'Đèn ông sao', 'Quả dâu tây', 'Cây thần', 'Quả dưa hấu'])
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
players_folder = 'players'
#CÁC HÀM CẦN THIẾT
#Khởi tạo các thông số trước để tránh lag
def create_game_folder(player_id):
    folder_name = os.path.join(players_folder, f"player_{player_id}")
    os.makedirs(folder_name, exist_ok=True)
    return folder_name
def save_frame(folder_name, frame):
    cv2.imwrite(os.path.join(folder_name, f"{player_id}_{datetime.now().strftime('%d_%H%M%S')}.png"), frame)
def save_game_result(folder_name, score, combo):
    with open(os.path.join(folder_name, f"{player_id}_{datetime.now().strftime('%d_%H%M%S')}.txt"), 'w') as result_file:
        result_file.write(f'Score: {score}\n')
        result_file.write(f'Combo: {combo}\n')
def upload_folder_to_cloudinary():
    zip_filename = 'players_data.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, files in os.walk(players_folder):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, players_folder)  # Add relative path in zip
                zipf.write(full_path, arcname=arcname)

    # Upload the zip file to Cloudinary
    upload_result = cloudinary.uploader.upload(zip_filename, resource_type="raw",public_id=f"players_data_{datetime.now().strftime('%d_%H')}")
    print("Uploaded players folder:", upload_result["secure_url"])

    # Cleanup: remove the zip file after uploading
    os.remove(zip_filename)
    os.removedirs(players_folder)
    
def init_game():
    global folder_name,took_pic,took_point,emo_id, emo_list, emo_pos, eraser, pen, xp, yp, is_saved, is_draw, is_spam, is_play, col, brush_size, canvas, start_point, end_point, result_icon, display_time, total_time, start_time, frame_count, target, target_pos, target_id, count, score, combo, correct
    GAME_SIZE=30
    folder_name = create_game_folder(player_id)
    took_pic=False
    took_point=False
    emo_id=np.random.choice(len(classes),GAME_SIZE)
    emo_list=[classes[emo] for emo in emo_id]
    emo_pos = [len(emo)*12 for emo in emo_list]
    eraser, pen = tools[0], tools[1]
    xp, yp = 0, 0 # tọa độ ngón tay trước đó để vẽ thành các nét line siêu ngắn (tăng độ mượt)
    is_saved = False #flag để tránh check tay mở liên tục
    is_draw = False #flag xác nhận người chơi đã vẽ j đó trc khi nộp
    is_spam = True 
    is_play = False  #flag xác nhận bắt đầu trò chơi
    col = (0, 255, 255)  # Default color (yellow)
    brush_size=25 #độ lớn cọ vẽ ban đầu
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8) #ảnh sẽ được vẽ lên đây
    start_point = (CENTER[0]-BOX_RANGE, CENTER[1]-BOX_RANGE)
    end_point = (CENTER[0]+BOX_RANGE, CENTER[1]+BOX_RANGE)    
    result_icon = None #hình vẽ dự đoán được mỗi lần nộp
    display_time = 0 #flag để cố định thời gian icon hiện lên
    total_time = 10 #tổng thời gian chơi
    start_time = 0 #thời gian bắt đầu chơi
    frame_count = 0
    target='' #hình vẽ mục tiêu
    target_pos=0
    target_id=0
    count=0
    score=int(0)
    combo=int(0)
    correct=int(0)
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
    pred_probab = model.predict(processed, verbose=0)[0]
    top_indices = np.argsort(pred_probab)[-3:][::-1]
    return top_indices
#upload ảnh lên cloud
def upload_image_to_cloudinary(image, image_name="predicted_image"):
    # Convert the image (box) to bytes in memory
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()

    # Upload the image bytes to Cloudinary
    response = cloudinary.uploader.upload(img_bytes, public_id=image_name, resource_type="image")
    # Get the uploaded image URL
    return response['secure_url']
    

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
    
def update_score(score,combo):
    score+=100*combo
    return score
def update(frame,start_time,total_time,score,combo):
    cv2.putText(frame, f"Score: {score}", (WIDTH-225, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"x{combo} Combo", (WIDTH-225, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.circle(frame, (50, 40), 30, (0,0, 255), 5)
    cv2.putText(frame, f"{int(total_time - (time.time() - start_time))}", (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
    
def main():
    global players,player_id,folder_name,took_pic, took_point, emo_id, emo_list, emo_pos, eraser, pen, xp, yp, is_saved, is_draw, is_spam, is_play, col, brush_size, canvas, start_point, end_point, result_icon, display_time, total_time, start_time, frame_count, target, target_pos, target_id, count, score, combo, correct
    init_game()
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image. Exiting.")
            break
        #Xử lý ảnh
        if is_play:
            if not took_pic:
                save_frame(folder_name, frame)
                took_pic=True
            start_time=time.time()
            combo=0
            score=0
            correct=0
            is_play=False
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=5)
        cv2.rectangle(frame,(LIMIT+OFFSET,HEIGHT//2-OFFSET*3-IMG_SIZE),(LIMIT+OFFSET+IMG_SIZE,HEIGHT//2-OFFSET*3),color=(0,0,0),thickness=5)
        cv2.rectangle(frame,(LIMIT+OFFSET,HEIGHT//2+OFFSET*3),(LIMIT+OFFSET+IMG_SIZE,HEIGHT//2+OFFSET*3+IMG_SIZE),color=(0,0,0),thickness=5)
            
        frame = putTextUnicode(frame, target, (CENTER[0]-target_pos,(CENTER[1]-BOX_RANGE)//5), font_path=font_path, font_size=font_size, color=(0, 0, 255))
        
        if start_time != 0: 
            current_time=int(time.time() - start_time)
            if current_time < total_time:
                update(frame,start_time,total_time,score,combo)
            elif current_time < total_time+10:
                if not took_point:
                    save_game_result(folder_name, score, correct)
                    took_point=True
                canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                cv2.putText(frame, f"Score: {score}", (WIDTH//2-150, HEIGHT//2-50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, f"Correct: {correct}", (WIDTH//2-150, HEIGHT//2+50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)      
            else:
                player_id+=1
                init_game()
        if is_spam:
            if count < len(emo_list): 
                target = emo_list[count]
                target_pos = emo_pos[count]
                target_id=emo_id[count]
                is_spam=False
            else:
                print("Reached the end of emo_list.")
                is_spam = False
            count+=1
        
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
                    class_label = keras_predict(model, box)
                    display_time = time.time()
                    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                    is_saved = True
                    is_draw=False
                    if target_id in class_label:
                        result_icon=emo[target_id]
                        is_spam=True
                        combo+=1
                        correct+=1
                    else: 
                        result_icon = emo[class_label[0]]
                        combo=0
                    score=update_score(score,combo)
                    #uploaded_image_url = upload_image_to_cloudinary(box, image_name=f"{classes[class_label[0]]}_{time.time()}")
                    #print(f"Uploaded image URL: {uploaded_image_url}")
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
        if result_icon is not None and time.time() - display_time < 1:
            overlay_icon(frame, result_icon)
        
        cv2.imshow('cam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): #Hủy game
            upload_folder_to_cloudinary()
            break
        if cv2.waitKey(1) & 0xFF == ord('n'): #Bắt đầu game/chuyển mục tiêu vẽ
            is_spam=True
        if cv2.waitKey(1) & 0xFF == ord('p'):
            is_play=True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
