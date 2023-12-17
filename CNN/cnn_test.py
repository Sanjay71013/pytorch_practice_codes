import os
import numpy as np
import glob
import pathlib
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from PIL import Image
from cnn_train import getClasses, ImageClassifier


def preprocessing():
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),  # 0-255 to 0-1 and numpy array to tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5],  # new_valuse = (old_value-mean)/sd
                             std=[0.5, 0.5, 0.5])  # 0-1 to [-1,1] and 2x3, first row is mean and 2nd row is standard deviation for all three channels
    ])
    return transform


def prediction(img_path, transform, classes):

    image = Image.open(img_path)

    image_tensor = transform(image).float()

    # Add a extra dimension for batch size
    image_tensor = image_tensor.unsqueeze_(0)
    if torch.cuda.is_available():
        print('Using GPU: ' + torch.cuda.get_device_name())
        image_tensor.cuda()

    input = Variable(image_tensor)
    output = model(input)
    index = output.data.numpy().argmax()
    pred = classes[index]

    return pred


if __name__ == "__main__":
    train_path = './data/train'
    pred_path = './data/pred'
    classes = getClasses(train_path=train_path)
    checkpoint = torch.load('best_checkpoint.model')
    model = ImageClassifier(num_classes=6)
    model.load_state_dict(checkpoint)
    model.eval()  # Set model in evaluation mode
    transform = preprocessing()
    img_path = glob.glob(pred_path+'/*.jpg')
    pred_dict = {}
    for i in img_path:
        pred_dict[i[i.rfind('/')+1:]] = prediction(img_path=i,
                                                   transform=transform, classes=classes)
    print(pred_dict)
