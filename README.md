# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#load dataset
df=pd.read_csv("CarPrice_Assignment.csv")

df.head()

#select feature and target
x=df[['enginesize','horsepower','citympg','highwaympg']] #numeric features only
y=df['price']

#split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#feature scaling
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#Train model
model=LinearRegression()
model.fit(x_train_scaled,y_train)

#prediction
y_pred=model.predict(x_test_scaled)

#model coefficients and metrics
#print("="*50)
print('Name: SUNDARESWARAN K')
print('Reg.No: 2122209050449')
print('MODEL COEFFICIENTS: ')
for feature, coef in zip(x.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")

print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test,y_pred):>10.2f}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(y_test,y_pred)):>10.2f}")
print(f"{'R-squared':>12}: {r2_score(y_test,y_pred):>10.2f}")
print(f"{'MAE':>12}: {mean_absolute_error(y_test,y_pred):>10.2f}")

# 1. Linearity Check
plt.figure(figsize=(12,5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linear Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

# 2. Independance (Durbin-Watson)
residuals=y_test-y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"""\nDurbin-Watson Statics: {dw_test:.2f}
(Values close to 2 indicate no autocorrelation)""")

# 3. Homoscedasticity
plt.figure(figsize=(12,5))
sns.residplot(x=y_pred,y=residuals,lowess=True,line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

# 4. Normality of residuals
fig, (ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title=("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:
<img width="203" height="153" alt="Screenshot 2026-02-04 090524" src="https://github.com/user-attachments/assets/c8605733-0a01-4f89-af6d-83703c277313" />

<img width="218" height="106" alt="Screenshot 2026-02-04 090530" src="https://github.com/user-attachments/assets/91952214-c940-4f19-aa8d-2809f42798cd" />

<img width="1012" height="460" alt="Screenshot 2026-02-04 090541" src="https://github.com/user-attachments/assets/ad04025a-b8ca-49f7-a7ce-6994e30f6392" />

<img width="898" height="393" alt="Screenshot 2026-02-04 090558" src="https://github.com/user-attachments/assets/f8555cd7-5ddf-44d7-a559-6ee9f21b912b" />

<img width="992" height="398" alt="Screenshot 2026-02-04 090607" src="https://github.com/user-attachments/assets/818f56b0-285c-4947-accb-37a3e2922b9c" />

## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
