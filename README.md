### DEVELOPED BY: SHRIRAM S
### REGISTER NO: 212222240098
### DATE:

# Ex.No: 07                                       AUTO REGRESSIVE MODEL

## AIM:
To Implement an Auto Regressive Model for Methi Price data using Python
## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model 
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
## PROGRAM
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/content/prices.csv', parse_dates=['Price Dates'], index_col='Price Dates')

print(data.head())

data= data[np.isfinite(data['Methi'])].dropna()

result = adfuller(data['Methi'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

model = AutoReg(train['Methi'], lags=13)
model_fit = model.fit()

predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test['Methi'], predictions)
print('Mean Squared Error:', mse)

plt.figure(figsize=(10,6))
plt.subplot(211)
plot_pacf(train['Methi'], lags=13, ax=plt.gca())
plt.title("PACF - Partial Autocorrelation Function")
plt.subplot(212)
plot_acf(train['Methi'], lags=13, ax=plt.gca())
plt.title("ACF - Autocorrelation Function")
plt.tight_layout()
plt.show()

print("PREDICTION:")
print(predictions)

plt.figure(figsize=(10,6))
plt.plot(test.index, test['Methi'], label='Actual pH value')
plt.plot(test.index, predictions, color='red', label='Predicted pH value')
plt.title('Test Data vs Predictions (FINAL PREDICTION)')
plt.xlabel('Date')
plt.ylabel('Methi Price')
plt.legend()
plt.show()

```
## OUTPUT:

### GIVEN DATA
![image](https://github.com/user-attachments/assets/c41477a5-71ad-4e6d-a52c-0dd9d1408394)

### ADF-STATISTIC AND P-VALUE
![image](https://github.com/user-attachments/assets/3873b5f9-0277-4ae6-8d71-3cc8b7613ad2)


### PACF - ACF
![image](https://github.com/user-attachments/assets/e8275d23-641a-4754-b040-fa53a654e405)

### MSE VALUE
![image](https://github.com/user-attachments/assets/bab10e03-ae89-413d-8625-296eed7b2456)



### PREDICTION
![image](https://github.com/user-attachments/assets/7441ca23-30c3-4d53-9f98-c2d4589940ae)

### FINAL PREDICTION
![image](https://github.com/user-attachments/assets/a88c899b-f91a-4615-b08f-78c5675e516a)


### RESULT:
Thus, the program to implement the auto regression function using python is executed successfully.
