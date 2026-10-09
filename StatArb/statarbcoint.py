# this is using ford and gm stock data

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# this is data collection + random walk
dates = pd.date_range(start='2025-01-01', periods=500, freq='D')
np.random.seed(42)

