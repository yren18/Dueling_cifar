import math
import numpy as np

def horizon_angle(coords):
    x1, y1, x2, y2 = coords
    dx, dy = x2 - x1, y2 - y1
    angle = math.atan2(dy, dx)
    return angle

def angle_to_degrees(angle):
    return math.degrees(angle)


def angle_to_so2(theta):
    """Return the SO(2) rotation matrix for an angle in (-pi, pi]."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    return R

def so2_to_angle(R, degrees=False):
    """Recover the angle θ from a 2 by 2 SO(2) rotation matrix."""
    theta = np.arctan2(R[1, 0], R[0, 0])  # in (-π, π]
    return np.degrees(theta) if degrees else theta


def so2_exp(omega):
    """Exponential map from tangent scalar ω → SO(2)."""
    c, s = np.cos(omega), np.sin(omega)
    return np.array([[c, -s],
                     [s,  c]])


def rotate_image(img, theta_deg):
    """Rotate image by theta degrees (positive = CCW)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), theta_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    return rotated