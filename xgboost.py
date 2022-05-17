import xgboost
print(xgboost.__version__)
from xgboost import XGBRegressor
# create an xgboost regression model
model = XGBRegressor(
    n_estimators=1000, 
    max_depth=7, 
    eta=0.1, 
    subsample=0.7, 
    colsample_bytree=0.8,
)
model.fit(X, y)
plt.scatter(X, y, color='teal', edgecolors='black', label='Train')
plt.plot(X, model.predict(X), color='orange',label='XGBoost regressor')
plt.title('XGBoost Regression')
plt.legend()
plt.show()