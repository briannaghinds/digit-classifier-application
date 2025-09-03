import streamlit as st
from model.mnist_model import MNIST_CNN
from model.utilities import train_model

class MNIST_Trainer():
    def __init__(self):
        st.set_page_config("MNIST Model Trainer", layout="centered")

    def page_build(self):
        st.title("MNIST Model Retraining")
        st.write("Here you can retrain the model once the dataset reaches a certain amount. Over time the model will learn the different ways each digit can be written and become more accurate.")


# MAIN
if __name__ == "__main__":
    trainer = MNIST_Trainer()
    trainer.page_build()