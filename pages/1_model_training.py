import streamlit as st
import pandas as pd
from model.mnist_model import MNIST_CNN
from utilities import train_model, test_model

# define image ID as state session variable


class MNIST_Training():
    def __init__(self):
        st.set_page_config("MNIST Training", layout="centered")

    def page_build(self):
        """
        FLOW
        - display the digit from .pt
        - display the guessed label
        - display the model's accuracy for it
        - have buttons where if the label is wrong correct it
        """
        if "visibility" not in st.session_state:
            st.session_state.visibility = "visible"
            st.session_state.disabled = False

        st.title("MNIST Model Trainer")
        st.write("Train the model!")
        # st.image("", caption="")
        st.write("IMAGE HERE")  # WILL DELETE LATER

        col1, col2 = st.columns(2)
        col1.write(f"Model's Guess: PREDICTION")
        col2.write(f"Model's Confidence XX%")

        st.write("Was the model wrong or correct in this instance?")
        col1_2, col2_2 = st.columns(2)
        correct = col1_2.button("Correct", key="disabled")
        wrong = col2_2.button("Wrong", key="visibility")

        if correct:
            print("YAY THE MODEL GUESSED CORERECT")

        # if wrong:
        # st.write("What is the correct label?")
        corrected_label = st.number_input(
            "What is the correct label?", 
            0, 
            9, 
            label_visibility="visible",
            disabled=st.session_state.disabled
        )
        st.write(corrected_label)
        print(f"CORRECTED LABEL: {corrected_label}")
        st.write(f"Image Label has been changed to: {corrected_label}")

        st.title("Updated MNIST Dataset")
        st.write("Displayed is a data editor where you can see the full DataFrame object and edit any labels if need be.")
        data = self.upload_mnist()
        edited = st.data_editor(data, width="stretch", num_rows="dynamic")


    def upload_mnist(self):
        mnist_df = pd.read_csv("./data/mnist.csv")
        print("MNIST Data Loaded")
        return mnist_df


    def data_refinement(self):
        pass

## MAIN
if __name__ == "__main__":
    train_page = MNIST_Training()
    train_page.page_build()