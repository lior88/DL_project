# DL project - Traffic sign recognition

<h1 align="center">
  <br>
Traffic sign recognition
  <br>
  <img src="https://github.com/lior88/DL_project/blob/main/source/pic.png" height="200">
</h1>
  <p align="center">
    <a href="https://github.com/lior88">Li-or Bar David </a> •
    <a href="https://github.com/kfirlevi"> Kfir Levi </a> 
  </p>

- [DL project - Traffic sign recognition](#dl-project---traffic-sign-recognition)
  * [Agenda](#agenda)
  * [Running The Code](#running-the-code)
    + [making the test set](#making-the-test-set)
    + [training and testing](#training-and-testing)
  * [The GTSRB dataset](#the-gtsrb-dataset)
  * [References](#references)


## Agenda

|File       | Topics Covered |
|----------------|---------|
|`main.py`| the main file of the project |
|`making_test_dataset.py`| arranging the test dataset as needed by the code |
|`Networks.py`| the implementations of the Original network and the Cifar_CNN network |


## Running The Code
### making the test set
before you use the main file, it is needed to arrange the test dataset in a specific way, the file making_test_dataset.py does it for you. it requires a single parameter:
* root - the location of the data folders.
#### run example:
    python making_test_dataset.py --root "C:\Users\liorb\OneDrive - Technion\Documents\Deep Learning - 046211\project"

### training and testing
In order to run the code, use the main.py file, it requires 4 parameters:
* TrainTest - receives string "Test" or "Train". use it to choose whether to train a model or test it.
* Classifier - receives a string which specifies the wanted Classifier. options are: 
  + Original - the original network used for the project.
  + CifarCNN - the Cifar_CNN network we saw in class.
  + resnet18 - the pretrained resnet18 network.
  + vgg16 - the pretrained vgg16 network.
  + densenet - the pretrained densenet network.
* Augnemtation - receives a string which specifies whether to use augmentation in the training or not.
* root - the location of the data folders.
#### run example:
    python main.py --TrainTest "Train" --Classifier "Original" --Augmentation "No" --root "C:\Users\liorb\OneDrive - Technion\Documents\Deep Learning - 046211\project"

## The GTSRB dataset
we used the GTSRB dataset, which consists of german traffic signs with 43 different classes and more than 39,000 samples of 30x30 RGB images.
in order to run the code, you need to download the dataset from one of the following sources:
  * https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
  * https://benchmark.ini.rub.de/gtsrb_dataset.html


## References
the original project, which we used as a base can be found at:
  * https://data-flair.training/blogs/python-project-traffic-signs-recognition/
