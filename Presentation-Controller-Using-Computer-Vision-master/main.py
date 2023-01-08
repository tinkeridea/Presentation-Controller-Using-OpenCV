import os
import cv2
from cvzone.HandTrackingModule import HandDetector


width, height = 1280, 720
folderPath = "presentation"
# Camera setup
cap = cv2.VideoCapture(2)
cap.set(3, width)
cap.set(4 ,height)


pathImages = os.listdir(folderPath)
# print(pathImages)

imgNumber = 3

hs, ws = int(120*1.2), int(213*1.2)
gestureThreshold = 300

buttonPressed = False
buttonCounter = 0
buttonDelay = 10
detector = HandDetector(detectionCon=0.8, maxHands=1)

annotations = []
annotationNumber = -1
annotationStart = False


scalingFactor = 1.5

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImages = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImages)


    hands, img = detector.findHands(img=img)

    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0,255,0), 10)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']

        lmList = hand["lmList"] # Finger landmark
        indexFinger = round(lmList[8][0] *scalingFactor * 2), round(lmList[8][1] * scalingFactor)


        # print(fingers)

        if cy <= gestureThreshold:
            if fingers == [1,0,0,0,0]:
                print("left")
                buttonPressed = True
                if imgNumber > 0:
                    annotations = []
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber -= 1
            
            if fingers == [0,0,0,0,1]:
                print("right")
                buttonPressed = True
                if imgNumber < len(pathImages)-1:
                    annotations = []
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber += 1

        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0,0,255), cv2.FILLED)

        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart= True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0,255,0), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)

        else:
            annotationStart = False


    if buttonPressed:
        buttonCounter +=1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0,0,200), 12)


    imgSmall = cv2.resize(img, (ws, hs))

    h, w, _ = imgCurrent.shape

    imgCurrent[0:hs, w-ws:w] = imgSmall

    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break