import cv2


def laplacian(image):
    ddepth = cv2.CV_16S
    kernel_size = 3
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(image_gray, ddepth, ksize=kernel_size)
    lap = cv2.convertScaleAbs(lap)
    return lap
    