import subprocess
import time

import cv2
import numpy as np


class Camera(object):
    def __init__(self, cap_id=0, int_matrix=None, dist_coef=None):
        self.cap_id = cap_id
        self.configure_v4l_camera()
        self.cap = cv2.VideoCapture(self.cap_id)

        self.configure_opencv_camera()

        time.sleep(0.25)

        self.ret, self.frame = self.cap.read()
        self.height, self.width, self.channels = self.frame.shape

        self.configure_v4l_camera()

        if int_matrix is not None:
            self.can_undistort = True
            self.M = int_matrix
            self.dist = dist_coef
            self.M_new, self.roi = cv2.getOptimalNewCameraMatrix(self.M, self.dist, (self.width, self.height), 1,
                                                                 (self.width, self.height))

        else:
            self.can_undistort = False
            self.M = None
            self.dist = None
            self.M_new, self.roi = None, None

    def configure_opencv_camera(self, width=640, height=480, fps=30, exposure_level=-4, auto_expo=0, no_v4l=False):
        print("Configuring OpenCV camera parameters...", end=" ")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_expo)
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_level)
        # self.cap.set(cv2.CAP_PROP_GAIN, -10)

        if no_v4l:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 1.0)
            self.cap.set(cv2.CAP_PROP_SATURATION, 1.0)
            self.cap.set(cv2.CAP_PROP_GAIN, 255)

        print("Done")

    def configure_v4l_camera(self, width=640, height=480, brightness=50, contrast=128, saturation=128, auto_wb=0,
                             gain=0, wb_temp=0, sharpness=0, exposure_mode=1, exposure_abs=15,
                             exposure_auto_priority=0, backlight_comp=0):

        cam_props = {'brightness': brightness, 'contrast': contrast, 'saturation': saturation,
                     'gain': gain, 'sharpness': sharpness, 'exposure_auto_priority': exposure_auto_priority,
                     'white_balance_temperature_auto': auto_wb, 'exposure_auto': exposure_mode,
                     'exposure_absolute': exposure_abs,
                     'white_balance_temperature_auto': auto_wb, 'white_balance_temperature': wb_temp}

        subprocess.call(
            ['sudo v4l2-ctl --set-fmt-video=width=' + str(width) + ',height=' + str(height) + ',pixelformat=YUYV'],
            shell=True)

        for key in cam_props:
            print(key, end='')
            subprocess.call(['sudo v4l2-ctl -d /dev/video' + str(self.cap_id) + ' --set-ctrl={}={}'.format(key, str(
                cam_props[key]))], shell=True)
            print()

    def read(self):
        self.ret, self.frame = self.cap.read()

    def frame(self):
        self.read()
        return self.frame()

    def frame_undistorted(self):
        self.read()

        if self.can_undistort:
            return cv2.undistort(self.frame(), self.M, self.dist, None, self.M_new)
        else:
            return self.frame

    def close(self):
        self.cap.release()


'''
*********************
V4l C270 Settings:
*********************
Brightness: 50 / 256
Contrast: 128 / 256
Saturation: 128 / 256
Auto WB: OFF
Gain: 0
White Balance Temp: 0
Sharpness: 0
Backlight Comp: 0
Exposure, Auto: Manual Mode
Exposure (Abs.): 128
Exposure, Auto Priority: OFF
'''


class C270(Camera):
    def __int__(self, cap_id=0):
        super().__init__(cap_id=cap_id,
                         int_matrix=np.array([[8.1226760328024022e+02, 0.0, 3.1595583251401598e+02],
                                              [0.0, 8.1536486059258914e+02, 2.2980426930795730e+02],
                                              [0.0, 0.0, 1.0]]),
                         dist_coef=np.array([[-7.7968254049911784e-02, 6.8336324508345037e-01,
                                              -5.7950590244645389e-03, 3.9975038267842673e-03,
                                              -1.7407537713228725e+00]]))
