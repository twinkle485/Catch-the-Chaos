import cv2
import mediapipe as mp
import time
import random
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, 
            max_num_hands=self.maxHands, 
            min_detection_confidence=self.detectionCon, 
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # Thumb and fingertip landmarks

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox
    
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers

def main():
    pTime = 0
    cap = cv2.VideoCapture(0) 
    detector = HandDetector()
    score = 0

    enemy_image = cv2.imread('enemy.png')  
    enemy_image = cv2.resize(enemy_image, (50, 50))  
    enemy_radius = 25

    x_enemy = random.randint(50, 600)
    y_enemy = random.randint(50, 400)
    enemy_speed_x = random.choice([-1, 1])  
    enemy_speed_y = random.choice([-1, 1])  

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        x_enemy += enemy_speed_x
        y_enemy += enemy_speed_y

        # Check for boundary collision
        if x_enemy <= enemy_radius or x_enemy >= (640 - enemy_radius):
            enemy_speed_x *= -1 
        if y_enemy <= enemy_radius or y_enemy >= (480 - enemy_radius):
            enemy_speed_y *= -1 

        # Calculate the top-left corner for the enemy image
        top_left_x = x_enemy - enemy_image.shape[1] // 2
        top_left_y = y_enemy - enemy_image.shape[0] // 2

        # Draw the enemy image
        img[top_left_y:top_left_y + enemy_image.shape[0], top_left_x:top_left_x + enemy_image.shape[1]] = enemy_image

        # Display score
        font = cv2.FONT_HERSHEY_SIMPLEX
        color_score = (255, 0, 0)  # Blue color for score
        img = cv2.putText(img, "Score: " + str(score), (10, 50), font, 1, color_score, 2, cv2.LINE_AA)

        # Check if hand landmarks are detected
        if len(lmList) != 0:
            index_finger_tip = lmList[8]  # Index finger tip
            cv2.circle(img, (index_finger_tip[1], index_finger_tip[2]), 25, (0, 200, 0), 5)

            # Check for collision with enemy
            if (index_finger_tip[1] >= x_enemy - 25 and index_finger_tip[1] <= x_enemy + 25) and \
               (index_finger_tip[2] >= y_enemy - 25 and index_finger_tip[2] <= y_enemy + 25):
                score += 1
                print("Found enemy!!!! Attack!!!!")
                x_enemy = random.randint(50, 600)
                y_enemy = random.randint(50, 400)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        color_fps = (255, 0, 0)  
        img = cv2.putText(img, f'FPS: {int(fps)}', (10, 100), font, 1, color_fps, 2, cv2.LINE_AA)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()