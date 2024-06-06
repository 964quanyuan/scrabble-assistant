import numpy as np
import cv2

def unwarp(image, corners):
    src_pts = np.float32(corners)
    dst_pts = np.float32([[0,0], [525,0], [525,525], [0,525]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    corrected_image = cv2.warpPerspective(image, matrix, (525, 525))
    cropped_image = corrected_image[0:525, 0:525]
    cropped_image = cv2.flip(cropped_image, 1)
    cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
    
    return cropped_image

def explode(image):
    fragments = []
    for x in range(15):
        for y in range(15):
            crop = image[x*35:(x+1)*35, y*35:(y+1)*35]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            fragments.append((crop, x, y))
    return fragments

def is_correct_shade_brown(tup):
    if sum(tup) <= 375:
        b = tup[0]
        g = tup[1]
        r = tup[2]
        rb = r / b if b != 0 else 0
        gb = g / b if b != 0 else 0
        rg = r / g if g != 0 else 0
        return (rb < 2.5 or rg < 2.5) and rb > 1.5 and gb < 1.45 and gb > 0.9
    return False

