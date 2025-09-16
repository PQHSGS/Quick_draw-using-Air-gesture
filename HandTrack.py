# handtrack.py
import math
import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplex,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.lmList = []

    def findHands(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if draw and self.results and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):
        self.lmList = []
        if not self.results or not self.results.multi_hand_landmarks:
            return self.lmList
        if handNo >= len(self.results.multi_hand_landmarks):
            return self.lmList
        myHand = self.results.multi_hand_landmarks[handNo]
        h, w, _ = img.shape
        for idx, lm in enumerate(myHand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            self.lmList.append([idx, cx, cy])
        return self.lmList

    def findDistance(self, p1, p2, img=None, draw=True, r=15, t=3):
        if len(self.lmList) <= max(p1, p2):
            return None, img, []
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        return length, img, [x1, y1, x2, y2, cx, cy]

    def fingersUp(self):
        # returns a 5-length list [thumb, index, middle, ring, pinky] where 1 means up
        if len(self.lmList) < 21:
            return [0, 0, 0, 0, 0]
        tipIds = [4, 8, 12, 16, 20]
        fingers = []
        # thumb (handedness not handled here; thumb test is heuristic)
        fingers.append(1 if self.lmList[tipIds[0]][1] < self.lmList[tipIds[0]-1][1] else 0)
        for i in range(1, 5):
            fingers.append(1 if self.lmList[tipIds[i]][2] < self.lmList[tipIds[i]-1][2] else 0)
        return fingers
