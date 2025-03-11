

# Import required libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

fish_df = pd.read_csv("Fish.csv")

# Display first few rows
fish_df.head()

# Check for missing values
print("Missing Values:\n", fish_df.isnull().sum())

# Encode categorical column 'Species' to numbers
label_encoder = LabelEncoder()
fish_df["Species"] = label_encoder.fit_transform(fish_df["Species"])

# Define features (X) and target variable (y)
X = fish_df.drop(columns=["Weight"])  # Features (all except Weight)
y = fish_df["Weight"]  # Target (Weight)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset shapes
print(f"Training Set: {X_train.shape}, Testing Set: {X_test.shape}")

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on training data
model.fit(X_train, y_train)

# Display the model coefficients & intercept
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Predict on test data
y_pred = model.predict(X_test)

# Compute evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Display performance metrics
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Save the trained model using pickle
model_filename = "fish_weight_model.pkl"

with open(model_filename, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved as {model_filename}")

# Load the saved model
with open(model_filename, "rb") as file:
    loaded_model = pickle.load(file)

# Test prediction with new data (Example: First row from test set)
sample_input = X_test.iloc[0].values.reshape(1, -1)  # Reshape for a single prediction
predicted_weight = loaded_model.predict(sample_input)

print(f"Predicted Fish Weight: {predicted_weight[0]:.2f} grams")