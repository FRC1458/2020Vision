import numpy as np
import cv2

from camera import Camera, C270
from vision_processing import RobotVision


if __name__ == '__main__':
    vis = RobotVision()
    vis.loop()
