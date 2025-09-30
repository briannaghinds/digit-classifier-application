## MNIST Classifier Application
This application is a Python project where an MNIST digit recognizer model is programmed to guess what a user types in a canvas object. The website is created/coded in `streamlit`, there are 4 different pages to this application:
1) The `home` page is where the user will write their digits and call the model to predict.
2) The `data labeling` page is where the user will go through the compiled digits written and make sure they are correctly labeled, this is the Human-In-The-Loop (HITL) part of this application.
3) The `model training` page is where after a certain amount of corrected digit labels (starting at 500) the user will be able to retrain the model to improve the overall accuracy of the model.
4) The `dashboard` page is where the data analytics is displayed.  

This project is actually aiding me in my capstone Data Analytics class where I am asking myself a question, analyzing data, and presenting my findings. The question I am asking is **What does model improvement look over time?** In this project I am not only building the data *to* retrain the model, but also looking at growth and improvment of the model per month.

### Libraries Used
These are the libraries defined with in the `requirements.txt` file.
```txt
os
streamlit
streamlit_drawable_canvas
cv2
datetime
matplotlib
csv
torch
torchvision
```
FORMAT LATER (for my BUSA 4360)

## PERSONAL NOTES:
- for the dashboard include graphs like:
*I want to have a compare/contrast display that has any digit (i.e. 3) and shows the month 1-month 3 difference of the low to high accuracy span.*
    - NONTECHNICAL
        - histogram/KDE plot grouped by months
        - digit accuracy heatmap (true label vs predicted label)
        - accuracy by digit pie chart
        - cumulative corrections over time(?)
        - confidence distribution per month
    - TECHNICAL
        - clustering graph
        - confidence distribution per digit
        - time series accuracy growth
- the graphs have to be in R so I can either import the graphs via R or I can call this library:
- I think it would be better to just use the csv file in RStudio and output the images there and upload it from here

```python
import rpy2.robjects as ro
ro.r('''
    library(ggplot2)
    df <- data.frame(x=rnorm(100), y=rnorm(100))
    p <- ggplot(df, aes(x, y)) + geom_point()
    ggsave("scatter.png", p)
''')
st.image("scatter.png")
```