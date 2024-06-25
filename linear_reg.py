import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('c:\\Users\\ajayakumar.vijayakum\\Desktop\\aj practise\\insurance.csv')

# Separate features and target variable
X = data.drop('charges', axis=1)
y = data['charges']

# Preprocess categorical variables using one-hot encoding
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create a pipeline with preprocessing and linear regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
model.fit(X, y)

# Print the coefficients
regressor = model.named_steps['regressor']
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)

# Predict charges for the entire dataset
predicted_charges = model.predict(X)

# Calculate the slope and intercept for the best fit line
m, b = np.polyfit(y, predicted_charges, 1)

# Plot actual vs. predicted charges
plt.figure(figsize=(10, 6))
plt.scatter(y, predicted_charges, alpha=0.5, label='Predicted vs. Actual')
plt.plot([y.min(), y.max()], [y.min() * m + b, y.max() * m + b], 'r-', lw=2, label=f'Best Fit Line: y = {m:.2f}x + {b:.2f}')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Fit Line: y = x')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs. Predicted Charges')
plt.legend()
plt.show()
