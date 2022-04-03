import ctypes
import math
import multiprocessing
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import win32api
import win32con


class HandDetector:
    def __init__(self, max_hands=2, detection_con=0.5, min_track_con=0.5):
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.min_track_con = min_track_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False, max_num_hands=self.max_hands,
                                        min_detection_confidence=self.detection_con,
                                        min_tracking_confidence=self.min_track_con)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

        self.results = None

    def find_hands(self, img, draw=True, flip_type=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        all_hands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                my_hand = {}
                my_lm_list = []
                x_list = []
                y_list = []
                for hid, lm in enumerate(handLms.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    my_lm_list.append([px, py])
                    x_list.append(px)
                    y_list.append(py)

                ## bbox
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                box_w, box_h = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, box_w, box_h
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                my_hand["lmList"] = my_lm_list
                my_hand["bbox"] = bbox
                my_hand["center"] = (cx, cy)

                if flip_type:
                    if handType.classification[0].label == "Right":
                        my_hand["type"] = "Left"
                    else:
                        my_hand["type"] = "Right"
                else:
                    my_hand["type"] = handType.classification[0].label
                all_hands.append(my_hand)

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, my_hand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return all_hands, img
        else:
            return all_hands

    def fingers_up(self, my_hand):
        my_hand_type = my_hand["type"]
        my_lm_list = my_hand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if my_hand_type == "Right":
                if my_lm_list[self.tipIds[0]][0] > my_lm_list[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if my_lm_list[self.tipIds[0]][0] < my_lm_list[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for hid in range(1, 5):
                if my_lm_list[self.tipIds[hid]][1] < my_lm_list[self.tipIds[hid] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        else:
            fingers = []
        return fingers

    @staticmethod
    def find_distance(p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info


def create_process(show_img: bool = True) -> multiprocessing.Process:
    temp = multiprocessing.Process(target=run, args=(show_img,), daemon=True)
    temp.start()
    return temp


def run(show_img: bool = True):
    cap = cv2.VideoCapture(0)
    w_cam = cap.get(3)
    h_cam = cap.get(4)
    detector = HandDetector(detection_con=0.8, max_hands=2)

    ploc_x, ploc_y = 0, 0
    smooth = True

    while True:
        success, img = cap.read()
        try:
            hands, img = detector.find_hands(img)
        except(Exception, KeyboardInterrupt):
            sys.exit()

        for hand in hands:
            if hand["type"] == "Right":
                lm_list1 = hand["lmList"]
                fingers = detector.fingers_up(hand)
                if fingers[1] == 1:
                    user32 = ctypes.windll.user32
                    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
                    x1 = lm_list1[8][0]
                    y1 = lm_list1[8][1]
                    x3 = np.interp(x1, (100, w_cam - 100), (0, screensize[0]))
                    y3 = np.interp(y1, (100, h_cam - 100), (0, screensize[1]))

                    if smooth is True:
                        smoothness = 2
                        cloc_x = ploc_x + (x3 - ploc_x) / smoothness
                        cloc_y = ploc_y + (y3 - ploc_y) / smoothness

                        x = int((screensize[0] - cloc_x))
                        y = int(cloc_y)

                        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        ploc_x, ploc_y = cloc_x, cloc_y
                    else:
                        x = int((screensize[0] - x3))
                        y = int(y3)

                    win32api.SetCursorPos((x, y))

                    if fingers[2] == 1:
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
                        time.sleep(0.2)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

            # if len(hands) == 2:
            #  # Hand 2
            #  hand2 = hands[1]
            #  lm_list2 = hand2["lmList"]  # List of 21 Landmark points
            #  bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            # center_point2 = hand2['center']  # center of the hand cx,cy
            # hand_type2 = hand2["type"]  # Hand Type "Left" or "Right"

            # fingers2 = detector.fingers_up(hand2)

            # Find Distance between two Landmarks. Could be same hand or different hands
            # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
            # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw

        if show_img:
            cv2.imshow("Image", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    run()
