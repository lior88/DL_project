import numpy as np
import numpy.matlib
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

class_names = ( 'Speed limit (20km/h)',
                'Speed limit (30km/h)',      
                'Speed limit (50km/h)',       
                'Speed limit (60km/h)',      
                'Speed limit (70km/h)',    
                'Speed limit (80km/h)',      
                'End of speed limit (80km/h)',     
                'Speed limit (100km/h)',    
                'Speed limit (120km/h)',     
                'No passing',   
                'No passing veh over 3.5 tons',     
                'Right-of-way at intersection',     
                'Priority road',    
                'Yield',     
                'Stop',       
                'No vehicles',       
                'Veh > 3.5 tons prohibited',       
                'No entry',       
                'General caution',     
                'Dangerous curve left',      
                'Dangerous curve right',   
                'Double curve',      
                'Bumpy road',     
                'Slippery road',       
                'Road narrows on the right',  
                'Road work',    
                'Traffic signals',      
                'Pedestrians',     
                'Children crossing',     
                'Bicycles crossing',       
                'Beware of ice/snow',
                'Wild animals crossing',      
                'End speed + passing limits',      
                'Turn right ahead',     
                'Turn left ahead',       
                'Ahead only',      
                'Go straight or right',      
                'Go straight or left',      
                'Keep right',     
                'Keep left',      
                'Roundabout mandatory',     
                'End of no passing',      
                'End no passing veh > 3.5 tons')


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
        epochs = 15
    elif opts.Classifier == "CifarCNN":
        model = CifarCNN()
        save_string = "cifar_cnn_ckpt"
        input_size = 30
        epochs = 15
    elif opts.Classifier == "resnet18":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  # replace the last FC layer
        input_size = 224
        save_string = "resnet_18_ckpt"
        epochs = 5
    elif opts.Classifier == "vgg16":
        model = models.vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)  # replace the last FC layer
        input_size = 224
        save_string = "vgg_16_ckpt"
        epochs = 5
    elif opts.Classifier == "alexnet":
        model = models.alexnet(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)  # replace the last FC layer
        input_size = 224
        save_string = "alexnet_ckpt"
        epochs = 5
    elif opts.Classifier == "densenet":
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)  # replace the last FC layer
        input_size = 224
        save_string = "densenet_ckpt"
        epochs = 5

    return model, save_string, input_size, epochs


num_classes = 43

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")  # use gpu 0 if it is available, o.w. use the cpu
print("device: ", device)

parser = argparse.ArgumentParser()
parser.add_argument('--TrainTest', type=str, default="Test", help="Train or Test")
parser.add_argument('--Classifier', type=str, default="Original", help="Original, CifarCNN, resnet18, vgg16, densenet, alexnet")
parser.add_argument('--root', type=str, help="directory of data folders")
opts = parser.parse_args()


model, save_string, input_size, epochs = model_choice(opts.Classifier, num_classes)
model = model.to(device)

mode = opts.TrainTest
root_train = opts.root + "/Train"
root_test = opts.root + "/Test_Arranged"


batch_size = 32
transform = transforms.Compose(
    [
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


data = torchvision.datasets.ImageFolder(root=root_train, transform=transform)
data_test = torchvision.datasets.ImageFolder(root=root_test, transform=transform)

train_size = int(0.8 * len(data))
val_size = len(data) - train_size
data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

if mode == "Train":

    learning_rate = 1e-3
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

    #%% plot the confusion matrix
    fig, ax = plt.subplots(1,1,figsize=(14,10))
    label_num = np.sum(confusion_matrix, axis=1)
    rep_label = numpy.matlib.repmat(label_num, len(class_names),1)
    Normalized_mat = np.round(confusion_matrix/rep_label, 2)
    # ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=500, cmap=plt.get_cmap('GnBu'))
    ax.matshow(Normalized_mat, aspect='auto', vmin=0, vmax=2, cmap=plt.get_cmap('GnBu'))
    

    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xlabel('Predicted Category')
    plt.ylabel('Actual Category')
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if round(confusion_matrix[i, j]/label_num[i], 2) != 0:
            #if confusion_matrix[i, j] != 0:
                text = ax.text(j, i, round(confusion_matrix[i, j]/label_num[i], 2),
                #text = ax.text(j, i, confusion_matrix[i, j],
                               ha="center", va="center", color="k")
    fig.tight_layout()
    plt.show()
    
elif mode == "Test":
    state = torch.load("./our_checkpoints/" + save_string + ".pth", map_location=device)
    model.load_state_dict(state['net'])

    test_accuracy, confusion_matrix = calculate_accuracy(model, test_loader, device)
    print("test accuracy: {:.3f}%".format(test_accuracy))
    
    #%% plot the confusion matrix
    fig, ax = plt.subplots(1,1,figsize=(14,10))
    label_num = np.sum(confusion_matrix, axis=1)
    rep_label = numpy.matlib.repmat(label_num, len(class_names),1)
    Normalized_mat = np.round(confusion_matrix/rep_label, 2)
    # ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=500, cmap=plt.get_cmap('GnBu'))
    ax.matshow(Normalized_mat, aspect='auto', vmin=0, vmax=2, cmap=plt.get_cmap('GnBu'))
    

    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xlabel('Predicted Category')
    plt.ylabel('Actual Category')
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if round(confusion_matrix[i, j]/label_num[i], 2) != 0:
            #if confusion_matrix[i, j] != 0:
                text = ax.text(j, i, round(confusion_matrix[i, j]/label_num[i], 2),
                #text = ax.text(j, i, confusion_matrix[i, j],
                               ha="center", va="center", color="k")
    fig.tight_layout()
    plt.show()

