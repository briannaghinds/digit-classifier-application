import streamlit as st

class MNIST_Dashboard():
    def __init__(self):
        st.set_page_config("MNIST Dashboard", layout="centered")

    def build_dashboard(self):
        st.write("BLAK BLAH BLAH")


## MAIN
if __name__ == "__main__":
    db = MNIST_Dashboard()
    db.build_dashboard()