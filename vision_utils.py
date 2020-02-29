import numpy as np
import cv2


def getRotation(contour):
    # This entire function borrowed from 3997 repo
    def translateRotation(rotation, width, height):
        if (width < height):
            rotation = -1 * (rotation - 90)
        if (rotation > 90):
            rotation = -1 * (rotation - 180)
            rotation *= -1
        return round(rotation)

    try:
        ellipse = cv2.fitEllipse(contour)
        centerE = ellipse[0]
        rotation = ellipse[2]
        widthE = ellipse[1][0]
        heightE = ellipse[1][1]
        rotation = translateRotation(rotation, widthE, heightE)
        return rotation
    except:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        center = rect[0]
        rotation = rect[2]
        width = rect[1][0]
        height = rect[1][1]
        rotation = translateRotation(rotation, width, height)
        return rotation


def get_distance(pixel_size, real_size=8.0, focal=8.1536486059258914e+02, scale=2.4):
    return scale * ((real_size * focal) / pixel_size)


def pose_3d(points_2d,
            camera_matrix=np.array([[8.1226760328024022e+02, 0.0, 3.1595583251401598e+02],
                                    [0.0, 8.1536486059258914e+02, 2.2980426930795730e+02],
                                    [0.0, 0.0, 1.0]]),
            dist_coeffs=np.array([[-7.7968254049911784e-02, 6.8336324508345037e-01,
                                   -5.7950590244645389e-03, 3.9975038267842673e-03,
                                   -1.7407537713228725e+00]])):
    # 3D model points (in)
    points_3d = np.array([
        # (x, y, z)
        (-9.75, 0.0, 0.0),  # Left Top
        (-5.0, -8.5, 0.0),  # Left Bottom
        (5.0, 8.5, 0.0),  # Right Bottom
        (9.75, 0.0, 0.0),  # Right Top
        (0.0, -4.25, 0.0)  # Center
    ])

    (success, rotation_vector, translation_vector) = cv2.solvePnP(points_3d, points_2d, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))

    return translation_vector, rotation_vector
