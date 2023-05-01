#framework streamlit
import streamlit as st

# for data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#for linear regression
from sklearn.linear_model import LinearRegression

#for MAE
from sklearn.metrics import mean_absolute_error

#--Read dataset--
df = pd.read_csv('NFLX Historical Data.csv')


#--preprocessing data--

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

#create new columns
new_index = []
for x in df.index:
    new_index.append(x)
print("finish")

#reverse columns
new_index.sort(reverse=True)

df['index'] = new_index
df = df.set_index('index')
df = df.rename_axis(None) #rename index become none
df = df.sort_index(ascending = True)



#--Create model liniear Regression--
valueTrain = ['Open','High','Low']
X = df[valueTrain]
y = df['Price']
#--split data --
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,test_size=0.4, shuffle=False)
# Define model
linearTrain = LinearRegression()
# Fit model
linearTrain.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = linearTrain.predict(val_X)

# PAGE 
st.set_page_config(page_title="Predict stocks",page_icon=":tada:",layout="wide")

# Import bootstrap
bootstrap = """
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-KK94CHFLLe+nY2dmCWGMq91rCGa5gtU4mk92HdvYe+M/SXH301p5ILy+dN9+nJOZ" crossorigin="anonymous">
</head>
"""
st.markdown(bootstrap, unsafe_allow_html=True)
st.markdown("""
    <nav class="navbar navbar-expand-lg bg-body-tertiary">
  <div class="container-fluid">
    <a class="navbar-brand" href="#">Navbar scroll</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarScroll" aria-controls="navbarScroll" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarScroll">
      <ul class="navbar-nav me-auto my-2 my-lg-0 navbar-nav-scroll" style="--bs-scroll-height: 100px;">
        <li class="nav-item">
          <a class="nav-link active" aria-current="page" href="#">Home</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="#">Link</a>
        </li>
        <li class="nav-item dropdown">
          <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown" aria-expanded="false">
            Link
          </a>
      </ul>
      <form class="d-flex" role="search">
        <input class="form-control me-2" type="search" placeholder="Search" aria-label="Search">
        <button class="btn btn-outline-success" type="submit">Search</button>
      </form>
    </div>
  </div>
</nav>

 """,unsafe_allow_html=True)

# Create a sidebar with a title and options
menu = ['Home', 'About', 'Contact']
choice = st.sidebar.selectbox("Select a page", menu)

# Create a page title and content based on the user's choice
if choice == 'Home':
    st.title('Home Page')
    st.write('Welcome to the home page.')
elif choice == 'About':
    st.title('About Page')
    st.write('This is the about page.')
else:
    st.title('Contact Page')
    st.write('You can contact us at contact@example.com')

# Dataset Container
with st.container():
    st.title("This is the dataset")
    st.table(df.head(10))

with st.container():
    st.write('---')
    left_visualization,right_visualization = st.columns(2)
    with left_visualization:
        st.header('Visualization 1')
        #Visualization
        testy_check = val_y.values.tolist()
        fig = plt.figure(figsize=(12, 6))
        #plt.figure(figsize = (12,6))
        plt.plot(testy_check , 'b' , label = 'Actual')
        plt.plot(val_predictions , 'orange' , label = 'Forecast')
        plt.grid()
        plt.legend()
        st.pyplot(fig)
    with right_visualization:
        st.header('Visualization 2')
        #Visualization
        testy_check = val_y.values.tolist()
        fig2 = plt.figure(figsize=(12, 6))
        #plt.figure(figsize = (12,6))
        plt.plot(testy_check , 'b' , label = 'Actual')
        plt.plot(val_predictions , 'orange' , label = 'Forecast')
        plt.grid()
        plt.legend()
        st.pyplot(fig2)


def predict(x1, x2, x3):
    # Load the saved model

    # Make the prediction
    prediction = linearTrain.predict([[x1, x2, x3]])

    return prediction[0]

# Use the user input to make the prediction

with st.container():
    st.write("---")
    inputData,Prediction = st.columns(2)
    with inputData:
        open = st.number_input('Enter your Open price')
        high = st.number_input('Enter your High price')
        low = st.number_input('Enter your Low price')
    with Prediction:
        # Use the inputs
        prediction = predict(open, high, low)
        st.header("Prediction")
        st.write("The prediction Is", prediction)