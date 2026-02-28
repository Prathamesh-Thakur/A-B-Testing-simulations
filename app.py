import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import scipy.stats as sst

import streamlit as st

# Page Configuration
st.set_page_config(page_title="A/B Testing Engine", layout="wide")

# Random number generator
rng = np.random.default_rng()

# The Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a Tool:", [
    "0. Data Generator",           
    "1. The Peeking Trap",
    "2. Dual-Engine Calculator",
    "3. Whale Bootstrapper",
    "4. Multi-Armed Bandit"
])

# Global Safety Net & Filters
# We only need the DataFrame for pages 1, 2, and 3.
if page in ["1. The Peeking Trap", "2. Dual-Engine Calculator", "3. Whale Bootstrapper"]:
    # The Check: Does the data exist in memory yet?
    if 'my_dataset' not in st.session_state:
        st.warning("⚠️ No data found! Please go to the '0. Data Generator' page to build your dataset first.")
        st.stop() # This halts the script here so charts don't throw errors
    
    # The Retrieval: Pull it from memory
    df = st.session_state['my_dataset']

    # The Sidebar Filters (Only render these if the data exists!)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Test Parameters")
    
    # Dynamically grab the variants that exist in the generated data
    available_variants = sorted(df['variant'].unique())
    control_group = st.sidebar.selectbox("Control Variant", available_variants, index=0)
    
    # Default treatment to the second variant if it exists, otherwise the first
    default_treatment_idx = 1 if len(available_variants) > 1 else 0
    variant_group = st.sidebar.selectbox("Treatment Variant", available_variants, index=default_treatment_idx)
    
    # Filter the dataset to ONLY the two selected variants
    df = df[df['variant'].isin([control_group, variant_group])]

# Page Routing
if page == "0. Data Generator":
    st.title("⚙️ Data Generation Playground")
    st.markdown("Adjust the levers below to simulate a custom A/B test.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Variant A")
        n_A = st.number_input("Visitors (A)", 1000, 50000, 10000)
        cr_A = st.slider("True Conv. Rate (A)", 0.01, 0.15, 0.04)
        
    with col2:
        st.subheader("Variant B")
        n_B = st.number_input("Visitors (B)", 1000, 50000, 10000)
        cr_B = st.slider("True Conv. Rate (B)", 0.01, 0.15, 0.05)
        
    with col3:
        st.subheader("Variant C")
        n_C = st.number_input("Visitors (C)", 1000, 50000, 10000)
        cr_C = st.slider("True Conv. Rate (C)", 0.01, 0.15, 0.03)

    if st.button("Generate Dataset", type="primary"):
        with st.spinner("Generating synthetic data..."):
            
            # --- Quick Vectorized Data Gen ---
            # Generate Variants
            variants = np.array(['A']*n_A + ['B']*n_B + ['C']*n_C)
            
            # Generate Conversions
            conv_A = rng.binomial(1, cr_A, n_A)
            conv_B = rng.binomial(1, cr_B, n_B)
            conv_C = rng.binomial(1, cr_C, n_C)
            conversions = np.concatenate([conv_A, conv_B, conv_C])
            
            # Generate Revenue (Only for converted users, average $50)
            revenue = conversions * rng.exponential(scale=50.0, size=len(conversions))
            
            # Generate Timestamps (Spread over 30 days)
            start_date = pd.to_datetime('2023-01-01')
            random_seconds = rng.integers(0, 30*24*60*60, size=len(conversions))
            timestamps = start_date + pd.to_timedelta(random_seconds, unit='s')
            
            # Assemble DataFrame
            new_df = pd.DataFrame({
                'timestamp': timestamps,
                'user id': np.arange(len(conversions)),
                'variant': variants,
                'conversion': conversions,
                'revenue': revenue
            })
            
            # Sort by time so the Peeking Trap page works correctly!
            new_df = new_df.sort_values('timestamp').reset_index(drop=True)
            
            # Save in memory
            st.session_state['my_dataset'] = new_df
            
        st.success("✅ Dataset successfully generated and locked into memory! You can now navigate to the analysis tools.")
        st.dataframe(new_df.head(), use_container_width=True)


elif page == "1. The Peeking Trap":
    st.title("The Peeking Trap: Frequentist Flaws")
    st.write("Tracking P-Values over time...")

    # Create date column
    df['date'] = df['timestamp'].dt.date
    
    # Get daily counts for each variant
    daily_stats = df.groupby(['date', 'variant']).agg(
        visitors=('user id', 'count'),
        conversions=('conversion', 'sum')
    ).reset_index()

    # Pivot the table so we have side-by-side columns for Control and Variant
    pivot_df = daily_stats.pivot(index='date', columns='variant', values=['visitors', 'conversions']).fillna(0)
    
    # Calculate Cumulative Sums (Running Totals)
    cum_df = pivot_df.cumsum()

    # Calculate daily P-values
    p_values_over_time = []
    dates = cum_df.index
    
    for date in dates:
        # Extract cumulative stats for the specific day
        ctrl_vis = cum_df.loc[date, ('visitors', control_group)]
        ctrl_conv = cum_df.loc[date, ('conversions', control_group)]
        var_vis = cum_df.loc[date, ('visitors', variant_group)]
        var_conv = cum_df.loc[date, ('conversions', variant_group)]
        
        # Prevent division by zero on early days
        if ctrl_vis < 10 or var_vis < 10 or (ctrl_conv + var_conv) == 0:
            p_values_over_time.append(1.0)
            continue
            
        ctrl_mean = ctrl_conv / ctrl_vis
        var_mean = var_conv / var_vis
        pooled_prob = (ctrl_conv + var_conv) / (ctrl_vis + var_vis)
        
        se = np.sqrt(pooled_prob * (1 - pooled_prob) * ((1 / ctrl_vis) + (1 / var_vis)))
        
        if se == 0:
            p_values_over_time.append(1.0)
        else:
            z = (ctrl_mean - var_mean) / se
            p_val = 2 * sst.norm.sf(abs(z))
            p_values_over_time.append(p_val)

    # 6. Plotting the results using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, p_values_over_time, marker='o', linestyle='-', color='blue', label='Daily P-Value')
    ax.axhline(y=0.05, color='red', linestyle='--', label='Alpha (0.05 Threshold)')
    
    # Formatting the chart
    ax.set_title('P-Value Fluctuation Over Time')
    ax.set_ylabel('P-Value')
    ax.set_xlabel('Date')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Render the chart in Streamlit
    st.pyplot(fig)

elif page == "2. Dual-Engine Calculator":
    st.title("Frequentist vs. Bayesian Engine")

    # Set alpha value
    alpha = 0.05
    
    # Get the converted and total visitor counts for each variant
    converted_counts = df[df['conversion'] == 1].groupby('variant', as_index = False).size()
    total_counts = df.groupby('variant', as_index = False).size()

    # Grab the relevant values according to the specified control and variant groups
    control_conv_tot = converted_counts[converted_counts['variant'] == control_group]['size'].values
    variant_conv_tot = converted_counts[converted_counts['variant'] == variant_group]['size'].values

    control_tot = total_counts[total_counts['variant'] == control_group]['size'].values
    variant_tot = total_counts[total_counts['variant'] == variant_group]['size'].values

    # Prior values for Bayesian testing
    control_alpha = 1
    control_beta = 1

    variant_alpha = 1
    variant_beta = 1

    col1, col2 = st.columns(2)

    # Run the z-test first

    with col1:
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

        st.metric(label = "Calculated p value:", value = p_value)

        # Reject or accept the null hypothesis
        if p_value < alpha:
            st.success("Statistically Significant! We have a winner.")

        else:
            st.error("Not Significant. The math says no.")
    
    # Run Bayesian test
    with col2:
        # Update alpha and beta values for both groups
        control_alpha += control_conv_tot
        control_beta += (control_tot - control_conv_tot)

        variant_alpha += variant_conv_tot
        variant_beta += (variant_tot - variant_conv_tot)
        
        # Sample size to compare
        sample_size = 100000

        # Values to be compared (Monte Carlo Simulation)
        control_values = rng.beta(a = control_alpha, b = control_beta, size = sample_size)
        variant_values = rng.beta(a = variant_alpha, b = variant_beta, size = sample_size)

        # Calculate how many times control had a smaller value than variant
        wins = (variant_values > control_values).sum()

        # Probability of variant winnnig
        probability = (wins / sample_size) * 100

        st.metric(label = "Probability that variant is better:", value = probability)

        # Plot the Bayesian curves

        # Get x values
        x_values = np.linspace(start = 0.00, stop = 1.00, num = 1000)
        
        # Get pdf values
        control_pdf = sst.beta.pdf(x = x_values, a = control_alpha, b = control_beta)
        variant_pdf = sst.beta.pdf(x = x_values, a = variant_alpha, b = variant_beta)

        # Create the plots
        fig, ax = plt.subplots(figsize = (10, 6))

        # Format x axis to show percentages
        formatter = mtick.PercentFormatter(xmax = 1.0)
        ax.xaxis.set_major_formatter(formatter)

        # Plot the curves
        ax.plot(x_values, control_pdf, color = 'blue', label = 'Control group', alpha = 0.4)
        ax.plot(x_values, variant_pdf, color = 'green', label = 'Variant group', alpha = 0.4)
        ax.fill_between(x_values, control_pdf, color = 'blue', alpha = 0.4)
        ax.fill_between(x_values, variant_pdf, color = 'green', alpha = 0.4)
        ax.legend()

        # Set limits on x and y axes view
        ax.set_xlim(0.00, 0.15)
        ax.set_ylim(bottom = 0)

        # Set axes labels
        ax.set_xlabel("Expected Conversion Rate")
        ax.set_ylabel("Probability Density")

        st.pyplot(fig)


elif page == "3. Whale Bootstrapper":
    st.title("The Whale Bootstrapper (Revenue)")

    # Get revenues for both control and variant groups
    control_revenues = df[df['variant'] == control_group]['revenue'].values
    variant_revenues = df[df['variant'] == variant_group]['revenue'].values

    # Total simulations for bootstrapping
    tot_simulations = 10000

    # Bootstrapping

    # Get the numbers for all simulations
    control_samples = rng.choice(control_revenues, size = ((tot_simulations, len(control_revenues))), replace = True)
    variant_samples = rng.choice(variant_revenues, size = ((tot_simulations, len(variant_revenues))), replace = True)
    
    # Calculate the means
    control_means = np.mean(control_samples, axis = 1)
    variant_means = np.mean(variant_samples, axis = 1)

    # Get the differences
    mean_differences = variant_means - control_means
    
    # Compute 2.5th and 97.5th percentile
    lower_percentile = np.percentile(mean_differences, 2.5)
    higher_percentile = np.percentile(mean_differences, 97.5)

    # Check if zero is between the interval 
    if lower_percentile < 0 and higher_percentile > 0:
        st.warning("We cannot say for sure!")

    elif higher_percentile < 0:
        st.error("Certified loser!")

    else:
        st.success("Certified winner!")

    # Define matplotlib figure object
    fig, ax = plt.subplots(figsize = (10, 7))

    # Plot histogram
    ax.hist(mean_differences, alpha = 0.4)
    
    # Add vertical lines at key points
    ax.axvline(x = 0, linestyle = 'dashed', color = 'red', label = 'Zero difference')
    ax.axvline(x = lower_percentile, linestyle = 'dashed', color = 'green', label = '2.5th percentile')
    ax.axvline(x = higher_percentile, linestyle = 'dashed', color = 'green', label = '97.5th percentile')

    # Set axes labels
    ax.set_xlabel("Differences between means")
    ax.set_ylabel("Frequency")
    ax.legend(loc = 'upper right')

    st.pyplot(fig)


elif page == "4. Multi-Armed Bandit":
    st.title("Thompson Sampling in Real-Time")
    st.write("Letting the algorithm find the winner dynamically...")
    
    # Simulated true conversion rates ['A', 'B', 'C']
    true_rates = [0.04, 0.09, 0.01]

    # Initial alpha and beta values for all three groups ['A', 'B', 'C']
    alpha_vals = [1, 1, 1]
    beta_vals = [1, 1, 1]

    # Total visitor count for each group
    total_counts = [0, 0, 0]

    # Number of simulations (or visitors)
    visitors = 10000

    # List to store the change in visitor counts
    history = []

    for _ in range(visitors):
        # Values to be compared (Monte Carlo Simulation)
        A_value = rng.beta(a = alpha_vals[0], b = beta_vals[0], size = 1)
        B_value = rng.beta(a = alpha_vals[1], b = beta_vals[1], size = 1)
        C_value = rng.beta(a = alpha_vals[2], b = beta_vals[2], size = 1)

        # Find the winning variation
        winner = np.argmax(np.array([A_value, B_value, C_value]))

        # Update total count for the winner
        total_counts[winner] += 1

        # Save the visitor counts for the variants
        history.append(total_counts.copy())

        # Simulating chance of conversion
        conversion = rng.binomial(n = 1, p = true_rates[winner])

        # Check if convered or not, and update alpha and beta values accordingly
        if conversion == 1:
            alpha_vals[winner] += 1
        else:
            beta_vals[winner] += 1
    
    actual_conversions = sum(alpha_vals) - 3

    # 2. Traditional A/B Test Conversions:
    # What if we forced an exact 33.3% even split for all 10,000 visitors?
    even_split = visitors / 3
    traditional_conversions = int((even_split * true_rates[0]) + (even_split * true_rates[1]) + (even_split * true_rates[2]))

    # 3. The Bandit Bonus:
    # The extra conversions we got because the Bandit starved the losers!
    conversions_saved = actual_conversions - traditional_conversions

    # Create three side-by-side columns
    col1, col2, col3 = st.columns(3)

    # Populate the columns with metrics
    col1.metric(label="Total Visitors", value=f"{visitors:,}")

    col2.metric(label="Total Conversions", value=f"{actual_conversions:,}")

    # The delta parameter adds a nice green up-arrow!
    col3.metric(
        label="Conversions Saved vs. A/B Test", 
        value=f"+{conversions_saved:,}", 
        delta="Bandit Bonus"
    )

    # Optional: Add a subtle divider before the chart
    st.divider()
    
    # Convert from list to numpy array
    history = np.array(history).astype(float)

    # Get the x axis values
    x_values = np.arange(1, visitors+1).astype(float)

    # Divide to get percentages
    history /= x_values[:, np.newaxis]

    fig, ax = plt.subplots(figsize = (10, 7))

    ax.stackplot(x_values, history[:, 0], history[:, 1], history[:, 2], labels = ['Variant A', 'Variant B', 'Variant C'], colors = ['blue', 'green', 'red'])
    ax.legend(loc = 'upper right')

    # Format x axis to show percentages
    formatter = mtick.PercentFormatter(xmax = 1.0)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_title("Multi-Armed Bandit Traffic Allocation")
    ax.set_xlabel("Number of Visitors")
    ax.set_ylabel("Percentage of Traffic")

    ax.set_xlim(left = 0, right = visitors)
    ax.set_ylim(bottom = 0.0, top = 1.0)

    st.pyplot(fig)
