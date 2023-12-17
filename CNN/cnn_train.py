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


def preprocessing(train_path, test_path):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 0-255 to 0-1 and numpy array to tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5],  # new_valuse = (old_value-mean)/sd
                             std=[0.5, 0.5, 0.5])  # 0-1 to [-1,1] and 2x3, first row is mean and 2nd row is standard deviation for all three channels
    ])
    # Dataloader to pass images in batches to the model for training
    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transform),
        batch_size=256, shuffle=True
    )
    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transform),
        batch_size=256, shuffle=True
    )
    return train_loader, test_loader


def getClasses(train_path):
    # Get the class names
    root = pathlib.Path(train_path)
    classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
    return classes


def model_train(model, epochs, loss_function, train_loader, test_loader, train_count, test_count):
    best_accuracy = 0.0
    print("----------STARTING TRAINING----------")

    for epoch in range(epochs):

        # Evaluation and training on training data
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data*images.size(0)
            _, prediction = torch.max(output.data, 1)

            train_accuracy += int(torch.sum(prediction == labels.data))

        train_accuracy = train_accuracy/train_count
        train_loss = train_loss/train_count

        # Evaluation on testing data
        model.eval()
        test_accuracy = 0.0
        for i, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            output = model(images)
            _, prediction = torch.max(output.data, 1)
            test_accuracy += int(torch.sum(prediction == labels.data))

        test_accuracy = test_accuracy/test_count

        print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss) +
              ' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))

        # Save the best model
        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), 'best_checkpoint.model')
            best_accuracy = test_accuracy


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ImageClassifier, self).__init__()
        # Input Shape is (256,3,150,150) = (batch_size, channel_size, Image_width (w), Image_height)
        # Output shape after Convolution is ((w-f+2P)/s)+1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape = (256,12,150,150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape = (256,12,150,150)
        self.relu1 = nn.ReLU()
        # Shape = (256,12,150,150)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the Image shape by a factor of 2
        # Shape = (256,12,75,75)

        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape = (256,20,75,75)
        self.relu2 = nn.ReLU()
        # Shape = (256,20,75,75)

        self.conv3 = nn.Conv2d(
            in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape = (256,32,75,75)
        self.relu3 = nn.ReLU()
        # Shape = (256,32,75,75)

        self.fc1 = nn.Linear(in_features=32*75*75, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.relu3(output)

        # Above output will be of shape (256,32,75,75)
        output = output.view(-1, 32*75*75)

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


if __name__ == '__main__':
    # Assigning the device type
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_path = './data/train'
    test_path = './data/test'
    train_loader, test_loader = preprocessing(
        train_path=train_path, test_path=test_path)
    classes = getClasses(train_path=train_path)
    model = ImageClassifier(num_classes=6).to(device=device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    num_epochs = 10
    train_count = len(glob.glob(train_path+'/**/*.jpg'))
    test_count = len(glob.glob(test_path+'/**/*.jpg'))
    model_train(model=model, epochs=num_epochs, loss_function=loss_function, train_loader=train_loader,
                test_loader=test_loader, train_count=train_count, test_count=test_count)
