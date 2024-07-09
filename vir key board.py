import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
from cvzone import cornerRect
from pynput.keyboard import Controller

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def drawAll(ListButtons, img):
    for button in ListButtons:
        x, y = button.pos
        h, w = button.size
        cornerRect(img, (x, y, w, h), 20, rt=0, colorC=(62, 10, 259))  # to draw the corner borders
        cv2.rectangle(img, button.pos, (x + w, y + h), (141, 90, 252), cv2.FILLED)  # bgr
        cv2.putText(img, button.text, (x + 15, y + 45), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    return img

keyboard = Controller()
detector = HandDetector(detectionCon=0.8, maxHands=2)

class Button:
    def __init__(self, pos, text, size=[60, 80]):
        self.pos = pos
        self.text = text
        self.size = size

keys = [["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        ["A", "Z", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["Q", "S", "D", "F", "G", "H", "J", "K", "L", "M"],
        ["W", "X", "C", "V", "B", "N", ",", ";", ":", "!"]]

ButtonList = []
for i in range(4):
    for j, key in enumerate(keys[i]):
        ButtonList.append(Button([100 * j + 100, 80 * i + 300], key))
ButtonList.append(Button([100, 80 * 4 + 300], " ", (60, 980)))
finalText = ""

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1200, 720))  # to resize the frame
    # detects the hand in the frame and locate the key points and draws the landmark
    hands, frame = detector.findHands(frame)
    img = drawAll(ButtonList, frame)

    if hands:  # here we check if the hand is detected or not
        for hand in hands:  # process each detected hand
            lmList = hand["lmList"]
            bbox = hand["bbox"]

            for button in ButtonList:
                x, y = button.pos
                h, w = button.size

                # Extract x and y coordinates of landmark number 8 and 12
                x1, y1 = lmList[8][0], lmList[8][1]
                x2, y2 = lmList[12][0], lmList[12][1]

                if x < x1 < x + w and y < y1 < y + h:  # check if your finger is in the range of the button
                    cv2.rectangle(frame, button.pos, (x + w, y + h), (199, 174, 254), cv2.FILLED)
                    cv2.putText(frame, button.text, (x + 15, y + 45), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
                    length, info = detector.findDistance((x1, y1), (x2, y2))
                    if length < 30:  # check the distance between the landmark 8 and the landmark 12
                        # keyboard.press(button.text)  # then open notepad and try typing with your virtual keyboard
                        cv2.rectangle(frame, button.pos, (x + w, y + h), (62, 10, 259), cv2.FILLED)
                        cv2.putText(frame, button.text, (x + 15, y + 45), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
                        finalText += button.text
                        sleep(1)

    cv2.rectangle(frame, (50, 50), (1100, 120), (199, 174, 254), cv2.FILLED)
    cv2.putText(frame, finalText, (60, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    # to find the landmarks
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
