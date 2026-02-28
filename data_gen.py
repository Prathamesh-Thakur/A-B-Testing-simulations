# Library imports
import numpy as np
import pandas as pd

# Random number generator
rng = np.random.default_rng()

# Number of visitors
num_visitors = 30000

# Time frame
num_days = 30

# TIMESTAMP SIMULATION
start_date = pd.to_datetime('2026-01-01 00:00:00') # start date
max_seconds = num_days * 24 * 60 * 60 # Calculate the maximum number of seconds in the time frame
random_seconds = np.sort(rng.integers(0, max_seconds, size=num_visitors)) # Generate random seconds and sort them so time flows forward
timestamps = start_date + pd.to_timedelta(random_seconds, unit='s') # Convert seconds to time and add to the start date

# USER IDS
user_ids = np.arange(num_visitors)

# VARIANT SIMULATION
# Generate variant to which user was routed
variants = np.random.choice(['A', 'B', 'C'], size = num_visitors)

# CONVERSION SIMULATION
# True conversion rates
conversion_rates = {'A': 0.04, 'B': 0.09, 'C': 0.01}

# To check p-hacking, use these conversion rates
# conversion_rates = {'A': 0.05, 'B': 0.05, 'C': 0.05}

# Actual conversion (0 or 1)
conversion = []

# Modeling conversion on the probability
for i in range(num_visitors):
    conversion.append(rng.binomial(n = 1, p = conversion_rates[variants[i]]))

# REVENUE SIMULATION
revenue = []

# Generate values only if converted
for i in range(num_visitors):
    if conversion[i] == 1:
        temp = 1500 + rng.pareto(a = 2, size = 1) * 10
        revenue.append(np.round(temp[0], decimals = 2))
    
    else:
        revenue.append(0.0)

# Append everything to a pandas dataframe and save the csv
data = pd.DataFrame(columns = ['timestamp', 'user id', 'variant', 'conversion', 'revenue'])
data['timestamp'] = timestamps
data['user id'] = user_ids
data['variant'] = variants
data['conversion'] = conversion
data['revenue'] = revenue

data.to_csv("data.csv", index = False)