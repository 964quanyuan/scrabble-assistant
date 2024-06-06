# Description: This file contains the functions that help find the precise corners of the image
# because the output of the keypoint detection model is only able to locate the corners
# within a roughly 10-pixel precision.

def calc_area(y, x, direction):
    if direction == 2:
        return x*y
    if direction == 3:
        return y*(50 - x)
    if direction == 0:
        return (50 - x)*(50 - y)
    if direction == 1:
        return x*(50 - y)
    
def is_correct_shade_gold(tup):
    b = tup[0]
    g = tup[1]
    r = tup[2]
    sumatrip = sum(tup)
    rb = r / b if b != 0 else 0
    gb = g / b if b != 0 else 0

    return rb < 1.76 and rb > 1.5 and gb < 1.45 and gb > 1.23 and sumatrip > 309

def find_corner(image, dir):
    curr_prod = 0
    corner = (0, 0)
    for x in range(50):
        for y in range(50):
            area = calc_area(y, x, dir)
            if area > curr_prod:
                if is_correct_shade_gold(image[y, x]):                
                    curr_prod = area
                    corner = (x, y)
    return corner 

def crop_corner(image, kpt_x, kpt_y):
    start_pt_x = max(kpt_x - 25, 0)
    start_pt_y = max(kpt_y - 25, 0)
    height, _, _ = image.shape
    if start_pt_y + 50 > height:
        start_pt_y = height - 50
        
    return (image[start_pt_y:start_pt_y+50, start_pt_x:start_pt_x+50], start_pt_x, start_pt_y)

def improve_corners(keypoints, image):
    new_kpts = []
    for idx, keypoint in enumerate(keypoints[0]):
        portion, global_x, global_y = crop_corner(image, keypoint[0], keypoint[1])
        local_x, local_y = find_corner(portion, idx)
        new_kpts.append([global_x + local_x, global_y + local_y])

    return new_kpts
        