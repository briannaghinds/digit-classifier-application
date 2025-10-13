import streamlit as st
from model.mnist_model import MNIST_CNN
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from model.utilities import train_model, test_model, SEPT_ACCURACY
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time


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

# MNIST data class
class MNISTCustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row["img_path"]
        label = int(row["true_label"])

        # open and preprocess
        img = Image.open(img_path).convert("L").resize((28, 28))
        img = np.array(img) / 255.0  # normalize 0-1
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # shape: (1,28,28)

        return img, label


class MNIST_Trainer():
    def __init__(self):
        st.set_page_config("MNIST Model Trainer", layout="centered")
        self.model = load_model()


    def mnist_data_refactor(self, df):
        # only use corrected data
        df = df[df["corrected"] == True].reset_index(drop=True)
        if df.empty:
            st.warning("No corrected data found. Please correct data in the trainer page first.")
            return None, None

        # shuffle + split
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

        # wrap in PyTorch datasets
        train_data = MNISTCustomDataset(train_df)
        print(train_data)
        test_data = MNISTCustomDataset(test_df)
        print(test_data)

        return train_data, test_data


    def model_retrain(self):
        # get the mnist.csv file, shuffle the data and then retrain the model
        mnist = pd.read_csv("./data/mnist.csv")
        train_data, test_data = self.mnist_data_refactor(mnist)
        
        # make sure there is data
        if train_data is None:
            return 

        # pass the data through train and test
        loaders = {
            "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0),
            "test": DataLoader(test_data, batch_size=100, shuffle=True, num_workers=0)
        }
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # retrain
        # EPOCHS = 5
        # for epoch in range(EPOCHS):
        #     # --- PROGRESS BAR UPDATE ---
        #     progress_percent = int((epoch / EPOCHS) * 100)
        #     progress_bar.progress(progress_percent, text=f"Training epoch {epoch+1}/{EPOCHS}...")

        #     # train model
        #     train_model(self.model, loaders["train"], 1, optimizer, loss_func, DEVICE)
        #     time.sleep(0.5)  # purely cosmetic to simulate work for UI smoothness

        # # Done training
        # progress_bar.progress(100, text="Training complete!")
        # time.sleep(1)
        # progress_bar.empty()
        train_model(self.model, loaders["train"], 15, optimizer, loss_func, DEVICE)
        st.success("Model finished retraining.")

        # get the accuracy
        accuracy_val = test_model(self.model, loaders["test"], loss_func, DEVICE)

        # if the accuracy is greater than the month before then replace 
        st.write(f"RETRAINING ACCURACY: {accuracy_val:.2f}")
        if accuracy_val > SEPT_ACCURACY:
            path = f".model/{accuracy_val:.2f}_MNIST_CNN_model.pth"
            # torch.save(self.model.state_dict(), path)
            st.success(f"Updated model saved to path: {path}")
        else:
            st.info("New model did not improve from before. Did not save.")


    def page_build(self):
        st.title("MNIST Model Retraining")
        st.write("Here you can retrain the model. Over time, the model will learn the different ways each digit can be written and become more accurate.")

        # retrain button will call the model_retrain() method
        if st.button("Retrain MNIST Model"):
            # call retrain function
            self.model_retrain()


# MAIN
if __name__ == "__main__":
    trainer = MNIST_Trainer()
    trainer.page_build()