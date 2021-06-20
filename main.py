import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
import time
import argparse

from Networks import CifarCNN, Original_Classifier


def calculate_accuracy(model, dataloader, device):
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([43, 43], int)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix


def model_choice(classifier, num_classes):
    if classifier == "Original":
        model = Original_Classifier()
        save_string = "original_classifier_ckpt"
        input_size = 30
    elif opts.Classifier == "CifarCNN":
        model = CifarCNN()
        save_string = "cifar_cnn_ckpt"
        input_size = 30
    elif opts.Classifier == "resnet18":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  # replace the last FC layer
        input_size = 224
        save_string = "resnet_18_ckpt"
    elif opts.Classifier == "vgg16":
        model = models.vgg16(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  # replace the last FC layer
        input_size = 224
        save_string = "vgg_16_ckpt"

    return model, save_string, input_size


num_classes = 43

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")  # use gpu 0 if it is available, o.w. use the cpu
print("device: ", device)

parser = argparse.ArgumentParser()
parser.add_argument('--TrainTest', type=str, default="Test", help="Train or Test")
parser.add_argument('--Classifier', type=str, default="Original", help="Original, CifarCNN, resnet18, vgg16")
parser.add_argument('--root', type=str, help="directory of data folders")
opts = parser.parse_args()

mode = opts.TrainTest
root_train = opts.root + "\Train"
root_test = opts.root + "\Test"

model, save_string, input_size = model_choice(opts.Classifier, num_classes)
model = model.to(device)

mode = opts.TrainTest
root_train = opts.root + "\Train"
root_test = opts.root + "\Test"

print(root_train)

batch_size = 32
transform = transforms.Compose(
    [
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# data = torchvision.datasets.DatasetFolder(root=root_train, loader=Image.open, extensions='.png', transform=transform)
data = torchvision.datasets.ImageFolder(root=root_train, transform=transform)

train_size = int(0.8 * len(data))
val_size = len(data) - train_size
data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True)
test_loader = val_loader

if mode == "Train":

    learning_rate = 1e-3
    epochs = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_memory = []
    for epoch in range(1, epochs + 1):
        model.train()  # put in training mode
        running_loss = 0.0
        epoch_time = time.time()
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            # send them to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # calculate the loss
            # always the same 3 steps
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters

            # print statistics
            running_loss += loss.data.item()

        # Normalizing the loss by the total number of train batches
        running_loss /= len(train_loader)

        # Calculate training/test set accuracy of the existing model
        train_accuracy, _ = calculate_accuracy(model, train_loader, device)
        val_accuracy, _ = calculate_accuracy(model, val_loader, device)

        log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Validation accuracy: {:.3f}% | ".format(epoch,
                                                                                                               running_loss,
                                                                                                               train_accuracy,
                                                                                                               val_accuracy)
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)
        loss_memory.append(running_loss)

        # save model
        if epoch % epochs == 0:
            print('==> Saving model ...')
            state = {
                'net': model.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('our_checkpoints'):
                os.mkdir('our_checkpoints')
            torch.save(state, "./our_checkpoints/" + save_string + ".pth")

    print('==> Finished Training ...')

    test_accuracy, confusion_matrix = calculate_accuracy(model, test_loader, device)
    print("test accuracy: {:.3f}%".format(test_accuracy))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
    plt.ylabel('Actual Category')
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
               '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
               '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
               '40', '41', '42',)
    plt.yticks(range(43), classes)
    plt.xlabel('Predicted Category')
    plt.xticks(range(43), classes)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(epochs), loss_memory)

    plt.show()

elif mode == "Test":
    state = torch.load("./our_checkpoints/" + save_string + ".pth", map_location=device)
    model.load_state_dict(state['net'])

    test_accuracy, confusion_matrix = calculate_accuracy(model, test_loader, device)
    print("test accuracy: {:.3f}%".format(test_accuracy))