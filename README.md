# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date:
# AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.
3. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
4. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.
5. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

# PROGRAM:
```
DEVELOPED BY: MANOJ G
REGISTER NO: 212222240060
```
```
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = '/mnt/data/astrobiological_activity_monitoring.csv'
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Extract the time series data (e.g., Atmospheric_Composition_O2)
time_series_data = data['Atmospheric_Composition_O2'].resample('D').mean().dropna()

# Plot ACF and PACF to visually inspect the data
plot_acf(time_series_data)
plot_pacf(time_series_data)
plt.show()

# Fit ARMA(1,1) model
model_arma_1_1 = ARIMA(time_series_data, order=(1, 0, 1))
arma_1_1_fit = model_arma_1_1.fit()

# Print the summary of the fitted ARMA(1,1) model
print(arma_1_1_fit.summary())

# Simulate an ARMA(1,1) process based on model parameters
ar1 = np.array([1, arma_1_1_fit.arparams[0]])
ma1 = np.array([1, arma_1_1_fit.maparams[0]])
ARMA_1_simulated = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)

# Plot the simulated ARMA(1,1) process
plt.plot(ARMA_1_simulated)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()

# Fit ARMA(2,2) model
model_arma_2_2 = ARIMA(time_series_data, order=(2, 0, 2))
arma_2_2_fit = model_arma_2_2.fit()

# Print the summary of the fitted ARMA(2,2) model
print(arma_2_2_fit.summary())

# Simulate an ARMA(2,2) process based on model parameters
ar2 = np.array([1, arma_2_2_fit.arparams[0], arma_2_2_fit.arparams[1]])
ma2 = np.array([1, arma_2_2_fit.maparams[0], arma_2_2_fit.maparams[1]])
ARMA_2_simulated = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)

# Plot the simulated ARMA(2,2) process
plt.plot(ARMA_2_simulated)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 200])
plt.show()

# Plot ACF and PACF for the ARMA(2,2) simulation
plot_acf(ARMA_2_simulated)
plot_pacf(ARMA_2_simulated)
plt.show()



```


# OUTPUT:
## SIMULATED ARMA(1,1) PROCESS:

![image](https://github.com/user-attachments/assets/6ad9e84a-90ee-49c8-934e-d5027e1f7639)


## Partial Autocorrelation

![image](https://github.com/user-attachments/assets/42e8f18a-e86b-4edb-9174-0dde26dc7de6)



## Autocorrelation

![image](https://github.com/user-attachments/assets/8eccf4e5-35d1-4dc8-afc3-095d1cff8b07)




## SIMULATED ARMA(2,2) PROCESS:

![image](https://github.com/user-attachments/assets/3ae4f09d-5586-44e2-a6d9-b7307b5f63a7)



## Partial Autocorrelation

![image](https://github.com/user-attachments/assets/5df368eb-64da-47b6-b531-e023d375b02f)


## Autocorrelation

![image](https://github.com/user-attachments/assets/060fe6a4-db52-4487-8c5c-57f0b1d27731)


# RESULT:
Thus,the python program is created to fit ARMA Model for Time Series successfully.
