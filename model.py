# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Sample dataset for AC sales prediction due to seasons
data = {
    'temperature': [30, 25, 40, 35, 45, 20, 10, 15, 50, 33],
    'season': ['Summer', 'Winter', 'Summer', 'Summer', 'Summer', 'Winter', 'Winter', 'Winter', 'Summer', 'Summer'],
    'sales': [500, 200, 600, 550, 700, 150, 100, 180, 800, 520]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert 'season' column to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['season'], drop_first=True)

# Split dataset into features and target
X = df.drop('sales', axis=1)
y = df['sales']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model to disk
with open('model.sav', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'model.sav'")
