import cv2

def match_resolution(img, sigma=1.0):
    """
    Applies Gaussian low-pass filtering to reduce effective-resolution mismatch before correlation.
    Preserves image dtype and range carefully.
    """
    # OpenCV calculates ksize automatically if passing (0,0) based on sigma
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return blurred
