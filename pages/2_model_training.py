import streamlit as st
from model.mnist_model import MNIST_CNN
from model.utilities import train_model, test_model, SEPT_ACCURACY
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


# global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    # load the model
    model = MNIST_CNN().to(DEVICE)
    print(model)
    model.load_state_dict(torch.load("./model/MNIST_CNN_model.pth"))
    model.eval()

    return model

class MNIST_Trainer():
    def __init__(self):
        st.set_page_config("MNIST Model Trainer", layout="centered")
        self.model = load_model()

    def page_build(self):
        st.title("MNIST Model Retraining")
        st.write("Here you can retrain the model. Over time, the model will learn the different ways each digit can be written and become more accurate.")

        # retrain button will call the model_retrain() method
        if st.button("Retrain MNIST Model"):
            # call retrain function
            self.model_retrain()


    def mnist_data_refactor(self, df):
        train_data = []
        test_data = []

        # split the data 70/30
        return train_data, test_data

        

    def model_retrain(self):
        # get the mnist.csv file, shuffle the data and then retrain the model
        mnist = pd.read_csv("./data/mnist.csv")

        # reformat the data, define train/test
        train_data, test_data = self.mnist_data_refactor(mnist)

        # define train/test data

        # pass the data through train and test
        loaders = {
            "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
            "test": DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
        }
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # retrain
        train_model(self.model, loaders["train"], 100, optimizer, loss_func, DEVICE)
        st.success("Model finished retraining.")


        # get the accuracy
        accuracy_val = test_model(self.model, loaders["test"], loss_func, DEVICE)

        # if the accuracy is greater than the month before then replace 
        if accuracy_val > SEPT_ACCURACY:
            path = f".model/{accuracy_val}_MNIST_CNN_model.pth"
            torch.save(self.model.state_dict(), path)
            st.success(f"Updated model saved to path: {path}")
        
        st.info("New model did not improve from before. Did not save.")

        pass


# MAIN
if __name__ == "__main__":
    trainer = MNIST_Trainer()
    trainer.page_build()