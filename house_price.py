import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("housing.csv")

# Features and target
X = data[["Size", "Bedrooms"]]
y = data["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
score = model.score(X_test, y_test)
print("Model Accuracy:", score)

# Predict custom input
size = int(input("Enter house size: "))
bedrooms = int(input("Enter number of bedrooms: "))

predicted_price = model.predict([[size, bedrooms]])
print("Estimated Price:", int(predicted_price[0]))

# Plot graph
plt.scatter(data["Size"], data["Price"])
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("House Price vs Size")
plt.show()
