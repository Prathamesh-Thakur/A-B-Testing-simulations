# A/B Testing Simulations

A comprehensive suite of Python-based simulations and interactive tools for understanding A/B testing methodologies, statistical pitfalls, and emerging bandit algorithms. This project demonstrates both classical and modern approaches to hypothesis testing through practical simulations and an interactive Streamlit dashboard.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Individual Modules](#individual-modules)
- [Dependencies](#dependencies)

## 🎯 Overview

This project illustrates critical concepts in A/B testing through hands-on simulations:
- **Statistical Foundations**: Learn how frequentist and Bayesian approaches differ
- **Common Pitfalls**: Understand p-hacking, peeking bias, and false positives
- **Advanced Techniques**: Explore Thompson sampling and bootstrap methods
- **Interactive Learning**: Use the Streamlit dashboard to experiment with different test scenarios

## ✨ Key Features

### 🚀 Interactive Streamlit Dashboard (`app.py`)
A user-friendly web interface with five integrated tools:

1. **Data Generator** - Create synthetic A/B test datasets with customizable parameters
   - Control visitor counts per variant (A, B, C)
   - Adjust true conversion rates
   - Generate realistic revenue distributions
   - Download and reuse generated datasets

2. **The Peeking Trap** - Demonstrate the dangers of continuous monitoring
   - Simulate repeated p-value calculations during a test
   - Visualize how p-values fluctuate over time
   - Measure the actual false positive rate from peeking
   - Compare against expected alpha level

3. **Dual-Engine Calculator** - Compare testing methodologies
   - **Frequentist (Z-test)**: Classical hypothesis testing with p-values
   - **Bayesian**: Probabilistic approach using beta distributions
   - Analyze the same dataset with both methods
   - Understand probability of variant superiority

4. **Whale Bootstrapper** - Handle outliers and non-normal distributions
   - Bootstrap resampling with confidence intervals
   - Analyze median differences between groups
   - Robust statistics for heavy-tailed distributions
   - Certification system for winners/losers

5. **Multi-Armed Bandit** - Thompson Sampling optimization
   - Dynamic allocation of traffic to variants
   - Exploration vs. exploitation tradeoff
   - Real-time probability updates with beta distributions
   - Optimal performance with minimal regret

### 📊 Purpose-Built Simulation Modules

Each module is designed for **individual experimentation** and can be run independently:

- **`multi-armed-bandit.py`** - Implements Thompson sampling algorithm for revenue optimization across multiple variants
- **`p-hacking.py`** - Simulates and quantifies the false positive rate caused by repeated testing and early stopping
- **`dual-engine-simulator.py`** - Compares Z-test vs. Bayesian inference on identical test data
- **`bootstrapper.py`** - Demonstrates bootstrap resampling for confidence interval estimation
- **`data_gen.py`** - Generates synthetic A/B test datasets with realistic timestamps, conversions, and revenue

## 🏗️ Project Architecture

```
A-B-Testing-simulations/
├── app.py                          # Main Streamlit dashboard
├── multi-armed-bandit.py           # Thompson sampling simulation
├── p-hacking.py                    # P-hacking false positive demo
├── dual-engine-simulator.py        # Z-test vs. Bayesian comparison
├── bootstrapper.py                 # Bootstrap resampling
├── data_gen.py                     # Synthetic data generation
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

## 💾 Installation

1. **Clone or extract the repository**
   ```bash
   cd A-B-Testing-simulations
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import streamlit; import pandas; import scipy; print('All dependencies installed!')"
   ```

## 🎮 Usage

### Running the Interactive Dashboard
```bash
streamlit run app.py
```
This launches the web interface where you can:
- Generate custom datasets interactively
- Run various A/B testing analyses
- Visualize results and statistics
- Compare different methodologies side-by-side

### Running Individual Experiments
Each module can be executed independently for focused exploration.

**Important**: Most modules require data, so **run `data_gen.py` first** to generate synthetic datasets before running other simulations:

```bash
# Step 1: Generate test datasets (run this first!)
python data_gen.py

# Step 2: Run individual experiments
# Demonstrate p-hacking effects
python p-hacking.py

# Compare testing methodologies
python dual-engine-simulator.py

# Explore Thompson sampling
python multi-armed-bandit.py

# Test bootstrap resampling
python bootstrapper.py
```

## 📁 Individual Modules

### **`app.py`** - Streamlit Dashboard
- **Type**: Interactive web application
- **Purpose**: Central hub for all A/B testing tools
- **Features**: Real-time data generation, variance testing, visualization, session state management
- **Run**: `streamlit run app.py`

### **`multi-armed-bandit.py`** - Thompson Sampling
- **Algorithm**: Bayesian Thompson Sampling
- **True Conversion Rates**: A=0.04, B=0.09, C=0.01
- **Simulation**: 10,000 visitor simulations
- **Output**: Traffic allocation per variant, winner identification
- **Key Insight**: Demonstrates optimal exploitation with adaptive learning

### **`p-hacking.py`** - P-hacking Simulation
- **Problem**: Continuous monitoring inflates false positive rates
- **Parameters**: 
  - Test Duration: 30 days
  - Daily Visitors: 100/group
  - True Conversion Rate: 0.05 (identical for both groups)
  - Simulations: 1,000 iterations
- **Output**: Expected vs. actual false positive rates (with visualization)
- **Key Insight**: Shows why early stopping increases Type I error

### **`dual-engine-simulator.py`** - Methodology Comparison
- **Methods**:
  1. Z-test (Frequentist): Classical hypothesis testing
  2. Bayesian: Beta-binomial conjugate prior approach
- **Test Data**: 430 conversions/10,000 visitors per group
- **Output**: P-value, significance determination, probability of variant superiority
- **Key Insight**: Illustrates philosophical differences between frameworks

### **`bootstrapper.py`** - Bootstrap Resampling
- **Distribution**: Pareto (heavy-tailed for outliers)
- **Sample Size**: 5,000 per group
- **Bootstrap Simulations**: 10,000 iterations
- **Confidence Level**: 95% (2.5th & 97.5th percentiles)
- **Output**: Winner certification or inconclusive result
- **Key Insight**: Robust statistics for skewed distributions

### **`data_gen.py`** - Data Generation Script
- **Features**:
  - 30,000 visitors across 30-day period
  - Three variants with different conversion rates
  - Realistic timestamps with sorting
  - Revenue distribution for converters (Pareto-based)
  - Optional configurations for p-hacking testing
- **Output**: Pandas DataFrame with full event logs
- **Customizable**: Conversion rates, visitor counts, timeframe

## 📦 Dependencies

```
scipy==1.17.1          # Statistical distributions and tests
pandas==2.3.3          # Data manipulation and analysis
streamlit==1.54.0      # Interactive web application framework
numpy                  # Numerical computations (implicit dependency)
matplotlib             # Visualization and plotting (implicit dependency)
```

## 🧠 Learning Outcomes

After exploring this project, you'll understand:

✅ **Statistical Foundations**
- Frequentist vs. Bayesian inference
- P-values and significance testing
- Type I error (false positives)

✅ **A/B Testing Pitfalls**
- Peeking bias and multiple testing problem
- Sample size and statistical power
- When to stop a test safely

✅ **Advanced Techniques**
- Thompson sampling for optimization
- Bootstrap resampling for robust estimates
- Handling outliers and non-normal distributions

✅ **Practical Implementation**
- Synthetic data generation
- Interactive visualization
- Real-world scenario modeling

## 🔬 Experimental Design

The project emphasizes **learning through experimentation**:
- Modify parameters in individual scripts to test hypotheses
- Use the dashboard to visually compare approaches
- Run standalone simulations for deep dives
- Iterate and observe how changes affect outcomes

---

**Note**: This codebase is designed for educational purposes. Each module is self-contained and can be modified or extended for custom experimentation.
