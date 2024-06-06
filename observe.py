import torch, cv2, queue, visualize, pickle, time #, tile_recognition_ml.dataset as dataset
import numpy as np, img_processing as process, precise_corners as polish
from solver.board import Board
from solver.dawg import Dawg, DawgNode
from concurrent.futures import ThreadPoolExecutor
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from torchvision import models
from torchvision.ops import nms
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from tile_recognition_ml.network import Network, classes
from sounds.play_audio import playsound, ramble

class MyEventHandler(FileSystemEventHandler):
    def __init__(self):
        with torch.no_grad():
            with ThreadPoolExecutor() as executor:
                future_krcnn = executor.submit(get_krcnn_model)
                future_ocr = executor.submit(get_ocr_model)
                future_device = executor.submit(torch.device, 'cuda')

                self.device = future_device.result()
                self.krcnn_model = future_krcnn.result()
                self.ocr_model = future_ocr.result()
            
            self.event_queue = queue.Queue()
            self.krcnn_model.eval()
            self.ocr_model.eval()
            self.krcnn_model.to(self.device)
            self.ocr_model.to(self.device)

    def on_created(self, event: FileSystemEvent):
        self.event_queue.put(event.src_path)

def get_krcnn_model(num_keypoints=4, weights_path='kpt_rcnn_ml/keypointsrcnn_biases.pth'):
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = models.detection.keypointrcnn_resnet50_fpn(weights_backbone=models.ResNet50_Weights.IMAGENET1K_V1,
                                                        num_keypoints=num_keypoints,
                                                        num_classes = 2, # Background is the first class, object is the second class
                                                        rpn_anchor_generator=anchor_generator)
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)        
        
    return model

def get_ocr_model():
    model = Network()
    model.load_state_dict(torch.load('tile_recognition_ml/fantastic_explosions.pth'))

    return model

def process_krcnn_output(output):
    scores = output[0]['scores'].detach().cpu().numpy()
    high_scores_idxs = np.where(scores > 0.7)[0].tolist() 
    post_nms_idxs = nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() 

    keypoints = []
    for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp[:2])) for kp in kps])

    bboxes = []
    for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        bboxes.append(list(map(int, bbox.tolist())))

    return bboxes, keypoints

def process_ocr_output(output):
    index = torch.argmax(output, dim=1).item()

    return classes[index]

def find_gameboard_corners(image):
    print("Working profusely to locate the gameboard corners...")

    tnsr_image = F.to_tensor(image).to(event_handler.device)
    output = event_handler.krcnn_model([tnsr_image])
    _, keypoints = process_krcnn_output(output)

    playsound('sounds/corners.wav')
    return polish.improve_corners(keypoints, image)

def filter_and_put_tiles(count, tiles, board):
    ramble(count % 4 + 1)
    for tile in tiles:
        tile_image = tile[0]
        tile_image = torch.from_numpy(np.array(tile_image, dtype=np.float32))
        tile_image = tile_image / 255.0
        tile_image = tile_image.unsqueeze(0)
        tile_image = tile_image.unsqueeze(1)
        tile_image = tile_image.to(event_handler.device)

        prediction = event_handler.ocr_model(tile_image)
        label = process_ocr_output(prediction)
        if "FALSE+" not in label:
            if label == "BLANK":
                letter = input(f"Enter the letter representing the blank at row #{tile[1]+1}, column #{tile[2]+1}: ")
                while not letter.isalpha():
                    playsound('sounds/bruh.wav')
                    letter = input(f"What's the letter for the blank at row #{tile[1]+1}, column #{tile[2]+1}: ")
                board.place_tile(letter.lower(), tile[1], tile[2])  
            else: 
                board.place_tile(label, tile[1], tile[2])

def get_board():
    with open("solver/csw.pickle", "rb") as input_file:
        word_graph = pickle.load(input_file)

    return Board(word_graph)

def get_rack():
    rack = input("Please enter your rack. Use '?' for blanks: ")
    while len(rack) != 7 or any((not char.isalpha()) and char != '?' for char in rack) or rack.count('?') > 2:
        playsound('sounds/bruh.wav')
        rack = input("Please enter your 7-letter rack. Use '?' for blanks: ")
    if rack.count('?') == 2:
        playsound('sounds/lucky.wav')
    else:
        playsound('sounds/juicy.wav')
    
    return rack

def get_plays(board, rack):
    valid_words = board.all_valid_plays(rack.upper())
    plays = board.sort_plays(valid_words)
    board.clear()
    print(plays[:100])

def display_loading_msg():
    print("loading program...")
    print("please wait patiently. it may take up to 15 seconds (or 15 minutes if ur on yo mama's laptop)")

def display_ready_msg():
    print("READY")
    print("press ctrl+S in the iVCAM window to take screenshot of the board")

if __name__ == "__main__":
    display_loading_msg()
    event_handler = MyEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path="C:\\Users\\31415\\Videos\\iVCam", recursive=True)
    observer.start()
    board = get_board()
    running = True
    count = 0
    display_ready_msg()
    
    try:
        while running:
            try:
                path = event_handler.event_queue.get(block=False, timeout=1)
                image = cv2.imread(path)

                precise_keypoints = find_gameboard_corners(image)
                visualize.visualize_keypoints(image.copy(), precise_keypoints)
                unwarped_image = process.unwarp(image, precise_keypoints)
                
                tiles = process.explode(unwarped_image)
                #dataset.save_frags(tiles, event_handler.device, event_handler.ocr_model, num)
                filter_and_put_tiles(tiles, board)
                board.display_board()

                rack = get_rack()
                start_time = time.time()
                get_plays(board, rack)
                print(f"Time taken: {time.time() - start_time}")
                count += 1

            except queue.Empty:
                pass
            
    except KeyboardInterrupt:
        running = False
        observer.stop()

    observer.join()