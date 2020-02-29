import numpy as np
import cv2

from camera import Camera, C270
from vision_utils import getRotation, get_distance, pose_3d

DEBUG = True
UNDISTORTED = True

ERODE = False
ERODE_KERNEL = np.ones((5, 5), np.uint8)

USE_APPROX_EPSILON = True
CNT_APPROX_EPSILON = 0.015

MAX_CNT_PTS = 10

MAX_CNT_AREA = 15000
MIN_CNT_AREA = 500

MAX_CNT_SOLIDITY = 0.25
MIN_CNT_SOLIDITY = 0.05


class RobotVision(object):
    def __init__(self, camera=C270(), hsv_low=np.array([20, 20, 70]), hsv_high=np.array([180, 255, 255])):
        self.camera: Camera = camera

        self.current_frame = None
        self.hsv = None
        self.mask = None

        self.running = True

        self.lower = hsv_low
        self.upper = hsv_high

        self.valid_targets = []

        if DEBUG:
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.namedWindow('hsv', cv2.WINDOW_NORMAL)

    def run(self):
        if UNDISTORTED:
            self.current_frame = self.camera.frame_undistorted()
        else:
            self.current_frame = self.camera.frame()

        if self.current_frame is None:
            return

        if ERODE:
            self.hsv = cv2.cvtColor(cv2.erode(self.current_frame, ERODE_KERNEL, iterations=1), cv2.COLOR_BGR2HSV)
        else:
            self.hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)

        self.hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv, self.lower, self.upper)

        self.valid_targets = []
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if USE_APPROX_EPSILON:
                epsilon = CNT_APPROX_EPSILON * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
            else:
                approx = cnt

            area = cv2.contourArea(approx)

            if area >= MAX_CNT_AREA or MIN_CNT_AREA > area:
                print("Area", area)
                continue

            hull = cv2.convexHull(approx)
            hull_area = cv2.contourArea(hull)

            solidity = float(area) / float(hull_area)

            if solidity >= MAX_CNT_SOLIDITY or MIN_CNT_SOLIDITY > solidity:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            rotation = getRotation(approx)
            dist = get_distance(w)

            M = cv2.moments(approx)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            leftmost = tuple(hull[hull[:, :, 0].argmin()][0])
            rightmost = tuple(hull[hull[:, :, 0].argmax()][0])
            topmost = tuple(hull[hull[:, :, 1].argmin()][0])
            bottommost = tuple(hull[hull[:, :, 1].argmax()][0])

            points_2d = np.array([
                leftmost,  # Left Top
                (399, 561),  # Left Bottom
                (337, 297),  # Right Bottom
                rightmost,  # Right Top
                (cx, cy)  # Center
            ], dtype="double")

            cv2.drawContours(self.current_frame, [hull], 0, (0, 0, 255), 2)

            cv2.putText(self.current_frame, str(dist), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 200, 235), 3,
                        cv2.LINE_AA)

            # self.valid_targets.append(approx)

        '''
        for n, cnt in enumerate(self.valid_targets):
            # (x, y), radius = cv2.minEnclosingCircle(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if len(cnt) >= 5:
                _, (MA, ma), angle = cv2.fitEllipse(cnt)
                cv2.putText(self.current_frame, str(int(MA)) + ', ' + str(int(ma)), (x + 20, y), cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (255, 200, 235), 3, cv2.LINE_AA)

            # center = (int(x), int(y))
            # radius = int(radius)

            # cv2.circle(self.current_frame, center, radius, (255, 0, 0), -1)
            cv2.rectangle(self.current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.current_frame, str(n), (x, y), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 200, 235), 3, cv2.LINE_AA)
        '''

    def loop(self):
        while self.running:
            self.run()
            self.debug()

    def debug(self):
        cv2.imshow("frame", self.current_frame)
        cv2.imshow("hsv", self.mask)

        wait_key = cv2.waitKey(1)

        if wait_key == ord('q'):
            cv2.destroyAllWindows()
            self.camera.close()

            self.running = False
