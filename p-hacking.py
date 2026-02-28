# Library imports
import numpy as np
import scipy.stats as sst
import matplotlib.pyplot as plt

# Set a true conversion rate for both control and variant groups
true_conversion_rate = 0.05

# Alpha value for testing
alpha = 0.05

# Test days
no_of_days = 30

# Daily visitors
no_of_control_visitors = 100
no_of_variant_visitors = 100

# Number of simulations
tot_simulations = 1000

# Track number of false positives
false_positives = 0

for _ in range(tot_simulations):
    # Cumulative conversion values
    control_conv_tot = 0
    variant_conv_tot = 0

    # Cumulative total values
    control_tot = 0
    variant_tot = 0

    # List of p values
    p_values = []

    for _ in range(no_of_days):
        # Simulate user behaviour for variant and control groups
        control_value = np.random.binomial(n = no_of_control_visitors, p = true_conversion_rate)
        variant_value = np.random.binomial(n = no_of_variant_visitors, p = true_conversion_rate)

        # Add them to total conversions
        control_conv_tot += control_value
        variant_conv_tot += variant_value

        # Maintain total number of visitors
        control_tot += no_of_control_visitors
        variant_tot += no_of_variant_visitors

        # Calculate mean conversion rates for both groups
        control_mean = control_conv_tot / control_tot
        variant_mean = variant_conv_tot / variant_tot
        
        # Pooled conversion probability
        pooled_prob = (control_conv_tot + variant_conv_tot) / (control_tot + variant_tot)

        # Standard error calculation
        standard_error = np.sqrt(pooled_prob * (1 - pooled_prob) * ((1 / control_tot) + (1 / variant_tot)))

        # Calculate z value
        z = (control_mean - variant_mean) / standard_error

        # Get critical z value
        p_value = 2 * sst.norm.sf(abs(z))

        # Append new p value to list
        p_values.append(p_value)

        if p_value < alpha:
            false_positives += 1
            break

    # Plot p values
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(p_values) + 1), p_values, marker='o', linestyle='-', color='b', label='Daily P-Value')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Alpha (0.05 Threshold)')
    plt.title('The Peeking Trap: P-Value Fluctuation Over 30 Days')
    plt.xlabel('Day of Test')
    plt.ylabel('Calculated P-Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Calculate the final False Positive Rate
actual_false_positive_rate = (false_positives / tot_simulations) * 100

print("-" * 30)
print(f"Expected False Positive Rate (Alpha): {alpha * 100}%")
print(f"Actual False Positive Rate due to Peeking: {actual_false_positive_rate:.1f}%")
