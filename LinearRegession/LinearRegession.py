import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('House_Rent_Dataset.csv')


st.title("Prediction of housing price")
st.table(df.head(4))

df = df.dropna() # Drop null-valued rows

x = np.array(df['Size'])
y = np.array(df['Rent'])

x_train = (x - np.min(x)) / (np.max(x) - np.min(x))
y_train = (y - np.min(y)) / (np.max(y) - np.min(y))

st.title("Rent vs size of room")
fig, axes = plt.subplots(1, 1, figsize=(8, 6))

# Use set_title instead of title
axes.set_title('Housing price Prediction')
axes.set_xlabel('size of the house in square foot')
axes.set_ylabel('Rent amount in Rupees in 10000')
axes.scatter(x_train, y_train, marker='*', c='r')

st.pyplot(fig)

shape_x = x_train.shape
shape_y = y_train.shape

shape_str = f"shape of training fetures:{shape_x[0]} and target:{shape_y[0]}"

st.markdown(shape_str)





def model(w,b,x_train,shape_x):
    Y = np.zeros(shape_x[0])
    for i in range(shape_x[0]):
        Y[i]= w*x_train[i]+b

    return Y


def calculate_cost(Y, y_train, shape_x):
    costs = np.zeros(shape_x[0])
    costs  = (Y - y_train)** 2
    cost = np.sum(costs)
    cost = cost / (2 * shape_x[0])
    return cost

def pdw(ypred,y,shape_x,x):
    p = np.zeros(shape_x[0])
    p=(ypred - y)*x
    pw = np.sum(p)/shape_x[0]
    return pw

def pdb(ypred,y,shape_x):
    p = np.zeros(shape_x[0])
    p =ypred- y
    return np.sum(p)/shape_x[0]


def Gradient(x, y, shape_x, a, num_iterations):
    W = 0
    B = 0
    for i in range(num_iterations):
        ypred = model(W, B, x, shape_x)
        J = calculate_cost(ypred, y, shape_x)
        pw = pdw(ypred, y, shape_x, x)
        tempW = W - a * pw
        pb = pdb(ypred, y, shape_x)
        tempB = B - a * pb

        W = tempW
        B = tempB



    return W ,B

# Adjust learning rate and number of iterations
W, B = Gradient(x_train, y_train, shape_x, 0.01, 100)





Y = model(W,B,x_train,shape_x)

st.title('Linear Regression :')

fig, axes = plt.subplots(1, 1, figsize=(8, 6))
# Use set_title instead of title
axes.set_title('Housing price Prediction')
axes.set_xlabel('size of the house in square foot')
axes.set_ylabel('Rent amount in Rupees in 10000')
axes.plot(x_train,Y,c='b',label = 'Predication')
axes.scatter(x_train, y_train, marker='*', c='r')

st.pyplot(fig)

st.title(f'w:{W}')
st.title(f'b:{B}')


ypred = model(W,B,x_train,shape_x)
J = calculate_cost(ypred,y_train,shape_x)
st.title(f"cost:{J}")

user_input = st.text_input("Enter your size of house in squarefoot")

if user_input:  # Check if the user has entered a value
    try:
        user_input = float(user_input)
        x_test = (user_input - np.min(x)) / (np.max(x) - np.min(x))
        Y = W * x_test + B
        Y = Y * ((np.max(y) - np.min(y)) + np.min(y))

        st.title(f'Your Home price: {Y}')
    except ValueError:
        st.error("Please enter a valid numeric value for the house price.")



