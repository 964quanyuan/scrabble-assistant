import cv2, torch, numpy as np
from tile_recognition_ml.network import classes

def explode(image):
    fragments = []
    for x in range(15):
        for y in range(15):
            crop = image[x*35:(x+1)*35, y*35:(y+1)*35]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            fragments.append(crop)
    return fragments

def process_ocr_output(output):
    index = torch.argmax(output, dim=1).item()

    return classes[index]

def save_frags(tiles, device, ocr_model, img_num):
    num = 1
    for tilly in tiles:
        tile = tilly[0]
        tile = torch.from_numpy(np.array(tile, dtype=np.float32))
        tile = tile / 255.0
        tile = tile.unsqueeze(0)
        tile = tile.unsqueeze(1)
        tile = tile.to(device)

        prediction = ocr_model(tile)
        label = process_ocr_output(prediction)
        cv2.imwrite(f"tile_recognition_ml/dataset/train/{label.lower()}/{img_num}-{num}.png", tilly[0])
        num += 1