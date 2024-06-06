import os, json, cv2, numpy as np, matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

from utils import collate_fn
from engine import train_one_epoch, evaluate

class ClassDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):                
        self.root = root
        self.transform = transform
        self.demo = demo
        self.annotations_file = "dataset\\train\\_annotations.coco.json"
    
    def __getitem__(self, idx):
        with open(self.annotations_file) as f:
            data = json.load(f)
            area = data['annotations'][idx]['area']
            bboxes_original = data['annotations'][idx]['bbox']
            keypoints_original = data['annotations'][idx]['keypoints']        
            img_path = data['images'][idx]['file_name']

        img_original = cv2.imread("dataset\\train\\" + img_path)
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
        return 336

KEYPOINTS_FOLDER_TRAIN = 'dataset/train'
KEYPOINTS_FOLDER_TEST = 'dataset/test'

dataset = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=None, demo=False)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

iterator = iter(data_loader)
batch = next(iterator)

print("Original targets:\n", batch[1], "\n\n")

def visualize(image, bboxes, keypoints):
    start_point = (bboxes[0][0], bboxes[0][1])
    end_point = (bboxes[0][2], bboxes[0][3])
    image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    for kps in keypoints:
        image = cv2.circle(image.copy(), tuple(kps), 1, (255,0,0), 10)

    plt.figure(figsize=(40,40))
    plt.imshow(image)
    plt.show()

image = (batch[0][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()
keypoints_xy = []
for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist()[0]:
    keypoints_xy.append(kps[:2])

visualize(image, bboxes, keypoints_xy)

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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_train = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=None, demo=False)
dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

data_loader_train = DataLoader(dataset_train, batch_size=3, shuffle=True, collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = get_model(num_keypoints = 4)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
num_epochs = 11

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device)
    
# Save model weights after training
torch.save(model.state_dict(), 'keypointsrcnn_biases.pth')
