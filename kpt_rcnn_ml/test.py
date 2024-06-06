import torch, json, cv2
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

class ClassDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):                
        self.root = root
        self.transform = transform
        self.demo = demo
        self.annotations_file = "dataset\\test\\_annotations.coco.json"
    
    def __getitem__(self, idx):
        with open(self.annotations_file) as f:
            data = json.load(f)
            area = data['annotations'][idx]['area']
            bboxes_original = data['annotations'][idx]['bbox']
            keypoints_original = data['annotations'][idx]['keypoints']        
            img_path = data['images'][idx]['file_name']

        img_original = cv2.imread("dataset\\test\\" + img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) 
        img, bboxes, keypoints = img_original, bboxes_original, keypoints_original   
        
        bboxes[2] = bboxes[0] + bboxes[2]
        bboxes[3] = bboxes[1] + bboxes[3]

        # Convert everything into a torch tensor        
        bboxes = torch.as_tensor([bboxes], dtype=torch.float32)       
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) # all objects are glue tubes
        target["image_id"] = idx
        target["area"] = torch.tensor([area])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = keypoints.view(1, 4, 3)     
        img = F.to_tensor(img)

        return img, target
    
    def __len__(self):
        return 15

dataset_test = ClassDataset('dataset/test', transform=None, demo=False)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, collate_fn=collate_fn)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
iterator = iter(data_loader_test)
images, targets = next(iterator)

tnsr_images = []
for image in images:
    print(image)
    image = image.to(device)
    tnsr_images.append(image)

def get_model(num_keypoints, weights_path=None):
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)        
        
    return model

with torch.no_grad():
    model = get_model(4, "keypointsrcnn_biases.pth")
    model.eval()
    model.to(device)
    output = model(tnsr_images)

print("Predictions: \n", output)

image = (tnsr_images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
scores = output[0]['scores'].detach().cpu().numpy()

high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

# Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
# Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
# Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

keypoints = []
for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
    keypoints.append([list(map(int, kp[:2])) for kp in kps])

bboxes = []
for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
    bboxes.append(list(map(int, bbox.tolist())))
    

def visualize(image, bboxes, keypoints):
    start_point = (bboxes[0][0], bboxes[0][1])
    end_point = (bboxes[0][2], bboxes[0][3])
    image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    for kps in keypoints[0]:
        image = cv2.circle(image.copy(), tuple(kps), 1, (255,0,0), 10)

    plt.figure(figsize=(40,40))
    plt.imshow(image)
    plt.show()

visualize(image, bboxes, keypoints)