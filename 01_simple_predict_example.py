import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#in this file, I show a simple linear regression estimator to predict a known arithmetic sequence

# make a fake dataset, using the known y = 2x 
data = {
    "x": np.array([1, 2, 3, 4, 5]),
    "y": np.array([2, 4, 6, 8, 10])
}
df = pd.DataFrame(data)

# split your data into features and target parameters, since there is only 2 here, x = feature, y = target 
x = df[["x"]]  # Sklearn expects a 2D array for features
y = df["y"]

# introduce the linear regression model 
model = LinearRegression()
model.fit(x, y)

# test the model and its prediction power with 6 (not from our dataset)
prediction = model.predict([[6]])

print("--- Model Verification ---")
print("DataFrame looks like...")
print(df)
print("Prediction for x = 6 is... ", prediction[0])

#now we want to apply this to NFL data to make a prediction model 