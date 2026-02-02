import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([52, 55, 61, 70, 82])

# Model A: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)

y_pred_linear = linear_model.predict(X)
mse_linear = mean_squared_error(y, y_pred_linear)
prediction_linear = linear_model.predict([[6]])

# Model B: Polynomial Regression (Degree 4)
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

y_pred_poly = poly_model.predict(X_poly)
mse_poly = mean_squared_error(y, y_pred_poly)
prediction_poly = poly_model.predict(poly.transform([[6]]))

# Output
print("MODEL A: Linear Regression")
print("Intercept:", linear_model.intercept_)
print("Coefficient:", linear_model.coef_[0])
print("Prediction at x=6:", prediction_linear[0])
print("Training MSE:", mse_linear)

print("\nMODEL B: Polynomial Regression (Degree 4)")
print("Prediction at x=6:", prediction_poly[0])
print("Training MSE:", mse_poly)
