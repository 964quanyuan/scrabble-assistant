import cv2
import numpy as np

# Global variables
dragging_kp = None
original_image = None
image = None

def mouse_callback(event, x, y, flags, params):
    global dragging_kp, original_image, image
    keypoints = params

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, kp in enumerate(keypoints):
            if np.sqrt((x - kp[0])**2 + (y - kp[1])**2) < 10:  # 10 is the radius of the circle
                dragging_kp = i
                break

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_kp is not None:
            keypoints[dragging_kp] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_kp = None

    image = original_image.copy()  # Create a fresh copy of the original image
    for i, kp in enumerate(keypoints):
        dest_pt = keypoints[(i + 1) % len(keypoints)]
        image = cv2.line(image, tuple(kp), tuple(dest_pt), (0,255,0), 2)
        image = cv2.circle(image, tuple(kp), 1, (255,0,255), 4)
        if i == 3:  # redraw first dot to make sure it is visible
            image = cv2.circle(image, tuple(keypoints[0]), 1, (255,0,255), 4)

def visualize_keypoints(input_image, keypoints):
    print("Found the corners! Let's fucking go!")
    print("Click and drag the magenta points if their positions look off. Press 'q' when you're done.")
    
    global original_image, image
    original_image = input_image
    image = original_image.copy()

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback, keypoints)

    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

def display_board(board):
    print("Oh shit! Here's the current board!")
    for row in board:
        print("|", end="")
        for square in row:
            if square:
                print(f" {square} ", end="|")
            else:
                print("   ", end="|")
        print("\n" + "-" * (4 * len(row) + 1))