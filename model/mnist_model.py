# torch libraries 
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

# # other data libraries
# import os
# import matplotlib.pyplot as plt
# import cv2

# global variables
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define the model architecture
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # define input, hidden, output
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 1 channel in, 10 out
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 10 channels in, 20 out
        self.conv2_dropout = nn.Dropout2d()  # dropout layer is a regualarization layer (randomly deactivates certain network nodes)
        self.fcl1 = nn.Linear(320, 50)
        self.fcl2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_dropout(self.conv2(x)), 2))

        # flatten data
        x = x.view(-1, 320)
        x = F.relu(self.fcl1(x))
        x = F.dropout(x, training=self.training)
        x = self.fcl2(x)

        return x
    
    def load_data(self):
        # # check if data/MNIST/raw folder exists
        # directory_exists = os.path.isdir("./data/MNIST/raw")
        # print(directory_exists)

        # if not directory_exists:
        # load the data
        train_data = datasets.MNIST(
            root="data",
            train=True,
            transform=ToTensor(),
            download=True
        )

        test_data = datasets.MNIST(
            root="data",
            train=False,
            transform=ToTensor(),
            download=True
        )

        # load the data into a DataLoader, turn into batches and shuffle
        loaders = {
            "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
            "test": DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
        }

        return loaders
    #     # else:
    #     #     return f"DATA ALREADY EXISTS"


    # def train_model(self, training_data, epochs, optimizer, loss_fuc, device):
    #     for epoch in range(epochs):
    #         self.train()

    #         for batch_i, (data, target) in enumerate(training_data):
    #             data, target = data.to(device), target.to(device)
    #             optimizer.zero_grad()

    #             output = self(data)
    #             loss = loss_fuc(output, target)
    #             loss.backward()
    #             optimizer.step()
                
    #         if batch_i % 20 == 0:
    #             print(f"Train Epoch: {epoch} [{batch_i * len(data)}/{len(training_data.dataset)} ({100. * batch_i / len(training_data):.0f}%)]\tLoss: {loss.item():.6f}")


    # def test_model(self, testing_data, loss_fuc, device):
    #     self.eval()

    #     test_loss = correct = 0

    #     with torch.no_grad():
    #         for data, target in testing_data:
    #             data, target = data.to(device), target.to(device)
    #             output = self(data)
    #             test_loss += loss_fuc(output, target).item()
    #             pred = output.argmax(dim=1, keepdim=True)
    #             correct += pred.eq(target.view_as(pred)).sum().item()

    #     test_loss /= len(testing_data)
    #     print(f"\nTest Set: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{len(testing_data)} ({100. * correct / len(testing_data):.0f}%\n)")
        


    # def predict_digit(self, img_tensor):
    #     output = self(img_tensor)
    #     prediction = output.argmax(dim=1, keepdim=True).item()

    #     probs = F.softmax(output, dim=1)
    #     confidence = probs.max().item()

    #     print(f"Prediction: {prediction}")
    #     return prediction, confidence


    # def save_model(self, path):
    #     torch.save(self.state_dict(), path)
    #     print(f"Model Saved to path: {path}")


    # def load_model(self, model_path):
    #     pass

# ## MAIN ##
# # define the model, optimizer model, loss function
# model = MNIST_CNN().to(DEVICE)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss_fuc = nn.CrossEntropyLoss()

# # upload the data
# loaders = model.load_data()

# # train the model, save it
# epochs = 11
# model.train_model(loaders["train"], epochs, optimizer, loss_fuc, DEVICE)

# # save the model
# model.save_model("MNIST_CNN_model.pth")