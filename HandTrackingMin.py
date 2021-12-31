import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
# use default parameters for .Hands(), we don't want to change anything
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                currX, currY = int(lm.x * w), int(lm.y * h)
                print(id, currX, currY)
                # if id == 0:
                cv2.circle(img, (currX, currY), 10, (81, 181, 248), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(
        img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (81, 181, 248), 3
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)

print("Code completed!")
