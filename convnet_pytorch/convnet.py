import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Mounting to google drive for running in Google Colab (free gpu's)

# from google.colab import drive
# drive.mount('/content/drive')
# root_path = '/content/drive/My Drive/Colab/'  # Google drive data folder

REBUILD_DATA = False
TRAIN_MODEL = True

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA: TRUE. Running on the gpu.")
else:
    device = torch.device("cpu")
    print("CUDA: FALSE. Running on the cpu")

class DogsVsCats():
    IMG_SIZE = 50
    CATS = "C:/Users/Dylan/Desktop/build/python_projects/pytorch_neural_network/convnet_pytorch/PetImages/Cat"
    DOGS = "C:/Users/Dylan/Desktop/build/python_projects/pytorch_neural_network/convnet_pytorch/PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    catcount = 0
    dogcount = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                # label is the directory PetImages/Cat
                # f is the filename in the directory label
                try:
                    path = os.path.join(label, f) # PetImages/Cat/ + cat_image_1.png
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # read each image in grayscale
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE)) # resize img to 50 x 50
                    # fill all image tensors to training data set (all cats, then all dogs)
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]]) # [[image], [1hot vector]]
                    
                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    # image being loaded (processed) is invalid
                    pass
                    print(str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats: ", self.catcount)
        print("Dogs: ", self.dogcount)

# Pre-process all images data into grayscale, count all image types and build training_data list
if REBUILD_DATA:
    dogsvscats = DogsVsCats()
    dogsvscats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle = True)

# plt.imshow(training_data[12][0], cmap = "gray") # training_data[] <- index of which file, [] <- index of image/label. [0] = image, [1] = label
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__() # init the nn.Module include
        self.conv1 = nn.Conv2d(1, 32, 5) # 5 kernel size (5x5 window)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50) # 50 x 50 images for input
        self._to_linear = None
        # Convs runs the first 3 convolutional layers to determine the data size, then we can use the linear (dense) layers
        # with a definitive output size.
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    # Convolutional layers to initialize convolutional data tensor size with random data
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # by now, x is the final output tensor from the conv layers.

        #print(x[0].shape)
        # Determine the needed tensor dimension for the fc1 layer input.
        if self._to_linear is None: # First initialization
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2] # 5 x 3 x 10 tensor
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear) # Flatten the input data (x) after its been passed through conv layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim = 1)

net = Net().to(device)

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50) # X is the tensor of 50 x 50 image pixels.
X = X / 255 # Scale 0-255 values between 0-1
y = torch.Tensor([i[1] for i in training_data]) # y is the tensor of labels [cat, dog]

VAL_PCT = 0.1
val_size = int(len(X) * VAL_PCT)
print(val_size)

train_X = X[:-val_size] # take front of list to -val_size (start at end, move back to val_size) (front majority of list)
train_y = y[:-val_size]

test_X = X[-val_size:] # -val_size (backward from end to val_size) onward (end portion of the list)
test_y = y[-val_size:]

BATCH_SIZE = 100
EPOCHS = 3

def train(net):
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    loss_function = nn.MSELoss()
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # step through len by BATCH_SIZE (100)
            #print(f"{i}:{i+BATCH_SIZE}")
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50) # batch of 100 image pixel tensors
            batch_y = train_y[i:i + BATCH_SIZE] # batch of 100 labels

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Begin fitment of model using each batch, backpropagate and adjust weights.
            # Need to zero the gradients in each backpropagation
            net.zero_grad()

            outputs = net(batch_X) # feedforward, gives forward end output
            loss = loss_function(outputs, batch_y) # loss(input, target) -> Value to be tested, correct target loss is determined from.
            loss.backward() # calculate loss on each weight
            optimizer.step() # adjust each weight

        print(f"\nEpoch: {epoch}. Loss: {loss}")

def test(net):
    correct = 0
    total = 0
    with torch.no_grad(): # Weights don't want to be adjusted (w/ gradient) at all during test data
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0] # 1 hot vector, 2 values (dog or cat)
            predicted_class = torch.argmax(net_out) # dog or cat

            if predicted_class == real_class:
                correct += 1
            total += 1

    print("\nAccuracy: ", round(correct / total, 3))

train(net)
test(net)
