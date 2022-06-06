import cv2
import numpy as np
import time
import os
import handtracking_module as htm

pTime = 0
cTime = 0
######################
brushThickness = 15
eraserThickness = 100
######################


folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (0,0,255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon = 0.65, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
#imgCanvas.fill(255)

while True:
    # import img
    success, img = cap.read()
    img = cv2.flip(img,1)
    
    # find the handlandmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList)!=0:
        #print(lmList)
        
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:] # for index finger
        x2, y2 = lmList[12][1:] # for middle finger
    
        # check fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
    
        # If selection mode - 2 fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            #print("selection mode")
            #checking for the click
            if y1 < 125:
                if 350<x1<450:
                    header = overlayList[0]
                    drawColor = (0,0,255)
                elif 480<x1<530:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 560<x1<610:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif 640<x1<690:
                    header = overlayList[3]
                    drawColor = (255,0,255)
                elif 900<x1<950:
                    header = overlayList[4]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1,y1-15), (x2,y2+15), drawColor, cv2.FILLED)
    
        # if drawing mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 7, drawColor, cv2.FILLED)
            #print("Drawing mode")
            if xp==0 and yp==0:
                xp, yp = x1, y1
            
            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp), (x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, eraserThickness)  
            else:
                cv2.line(img, (xp,yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, brushThickness)
                
            xp, yp = x1, y1
    
    
    
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
    
    # Setting the header img
    img[0:125, 0:1280] = header
    
    #fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (1200,700), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    #img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    # cv2.imshow("Canvas",imgCanvas)
    # cv2.imshow("Inv",imgInv)
    cv2.waitKey(1)