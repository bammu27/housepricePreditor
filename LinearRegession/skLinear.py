import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('House_Rent_Dataset.csv')

st.title("Prediction of housing price")
st.table(df.head(4))

# Drop null-valued rows
df = df.dropna()

# Extract features (Size) and target (Rent)
x = df[['Size']]
y = df['Rent']

# Create and fit a Linear Regression model
model = LinearRegression()
model.fit(x, y)

st.title("Rent vs size of room")
fig, axes = plt.subplots(1, 1, figsize=(8, 6))

# Use set_title instead of title
axes.set_title('Housing price Prediction')
axes.set_xlabel('size of the house in square foot')
axes.set_ylabel('Rent amount in Rupees in 10000')
axes.scatter(x, y, marker='*', c='r')

# Generate predictions
y_pred = model.predict(x)
axes.plot(x, y_pred, color='blue', linewidth=2, label='Prediction')

st.pyplot(fig)

# Get the model coefficients and intercept
W = model.coef_[0]
B = model.intercept_

st.title(f'w:{W}')
st.title(f'b:{B}')

# Calculate the cost (mean squared error)
cost = mean_squared_error(y, y_pred)
st.title(f"cost:{cost}")

st.title('Enter your home size:')
user_input = float(st.number_input("Enter the size of your home in square feet"))
if st.button("Predict Rent"):
    homeprice = W*user_input+B
    st.title(f"your home price:{homeprice}")