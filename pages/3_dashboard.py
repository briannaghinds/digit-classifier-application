import streamlit as st

class MNIST_Dashboard():
    def __init__(self):
        st.set_page_config("MNIST Dashboard", layout="centered")

    def build_dashboard(self):
        st.title("MNIST Analyst Dashboard")
        st.write("This projects purpose is to see the ")


## MAIN
if __name__ == "__main__":
    db = MNIST_Dashboard()
    db.build_dashboard()