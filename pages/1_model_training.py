import streamlit as st

# BUILD UP A TRAINING FOR MNIST


class MNIST_Training():
    def __init__(self):
        st.set_page_config("MNIST Training", layout="centered")

    def page_build(self):
        st.title("MNIST Model Trainer")
        st.write("Train the model!")


## MAIN
if __name__ == "__main__":
    train_page = MNIST_Training()
    train_page.page_build()