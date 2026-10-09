import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.core.window.doc import kwargs_numeric_only
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

dates = pd.date_range(start='2025-01-01', periods=500, freq='D')
np.random.seed(42)
Price_A = np.cumsum(np.random.randn(500)) + 100
Price_B = np.cumsum(np.random.randn(500)) + 100
df = pd.DataFrame({'Date':dates, 'Price_A':Price_A, 'Price_B':Price_B})
df.set_index('Date', inplace=True)

score, p_value, _ = coint(df['Price_A'], df['Price_B'])
print(f'cointegration test p val: {p_value}')
if p_value < 0.05:
    print("cointegrated, statarb valid")

X = sm.add_constant(df['Price_B'])
model = sm.OLS(df['Price_A'], X).fit() # linear regression
df['spread'] = df['Price_A'] - model.predict(X)

spread_mean = df['spread'].mean()
spread_std = df['spread'].std()
print(f'spread mean is {spread_mean}')
print(f'spread std dev is {spread_std}')

entry_threshold = spread_mean + 2 * spread_std
exit_threshold = spread_mean

initial_cash = 100000
portfolio_value = initial_cash
position_A = 0
position_B = 0

portfolio_values = []
entry_points = []
exit_points = []

for i in range(len(df)):
    if df['spread'].iloc[i] > entry_threshold:
        if position_A == 0 and position_B == 0:
            position_A = -1
            position_B = 1
            entry_points.append(df.index[i])
    elif df['spread'].iloc[i] < -entry_threshold:
        if position_A == 0 and position_B == 0:
            position_A = 1
            position_B = -1
            entry_points.append(df.index[i])
    elif df['spread'].iloc[i] < exit_threshold and (position_A != 0 or position_B != 0):
        if position_A != 0 and position_B != 0:
            position_A = 0
            position_B = 0
            exit_points.append(df.index[i])

portfolio_value += (position_A * df['Price_A'].iloc[i] + position_B * df['Price_B'].iloc[i])
portfolio_values.append(portfolio_value)

final_portfolio_value = portfolio_value
print(f'final portfolio val is: {final_portfolio_value}')

plt.figure(figsize=(10,6))
plt.plot(Price_A, Price_B, label="port. val.",linestyle='--') # got it to work but not what i want it to bep
plt.title('portfolio val. over time')
plt.xlabel('date')
plt.ylabel('port. val')
plt.legend()
plt.show()

# optional



