import multiprocessing as mp
import os
import numpy as np
import configparser

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import tqdm

from torchvision import models
from glob import glob

from PIL import Image
from tqdm import tqdm

config = configparser.ConfigParser()
config.read('./config/config.ini')
model_name = "model_ch3_bottom2"

transform = transforms.Compose([
    transforms.Resize(eval(dict(config[model_name])["transforms"])["resize"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=eval(dict(config[model_name])["transforms"])["normalize"][0],
                         std=eval(dict(config[model_name])["transforms"])["normalize"][1]),
])
idx_to_cls = eval(dict(config[model_name])["idx_to_cls"])
num_classes = int(dict(config[model_name])["num_classes"])

model = models.efficientnet_v2_l(weights=None)

model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
check_point = torch.load("./model_ch3_bottom2.pth")

for key in list(check_point.keys()):
    if "module." in key:
        check_point[key.replace("module.", "")] = check_point[key]
        del check_point[key]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(check_point)
model.to(device)
model.eval()

img_list = glob("./img/*.jpg")
predicted_list = []

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings=np.zeros((len(img_list), 1))

with torch.no_grad():
    for i, image_path in enumerate(tqdm(img_list, ncols=160, ascii=" =", unit="image")):
        starter.record()
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)

        predicted_list.append(predicted.item())
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[i] = curr_time

print(predicted_list)
print(timings)

# [ 11:46, 26.61 image/s, 43ms ]

