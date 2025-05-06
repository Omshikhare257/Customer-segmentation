# Customer Segmentation Analysis & Prediction

![Customer Segmentation](https://img.shields.io/badge/ML-Customer%20Segmentation-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.15+-red)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)

## 📊 Overview

A powerful Streamlit application that performs customer segmentation analysis using machine learning techniques. This tool helps businesses understand their customer base by identifying distinct customer segments and provides actionable marketing recommendations for each segment.


## ✨ Features

- **Data Analysis**: Upload and analyze your customer dataset with comprehensive EDA (Exploratory Data Analysis)
- **Automatic Feature Detection**: Automatically identifies numeric and categorical features
- **Missing Value Handling**: Options to handle missing data through various strategies
- **Customer Segmentation**: Uses K-means clustering to segment customers based on selected features
- **Interactive Visualizations**: 
  - Distribution plots for numeric features
  - Categorical data analysis
  - Correlation analysis with heatmaps and scatter plots
  - 2D and 3D cluster visualizations
- **Cluster Interpretation**: Generates human-readable descriptions of each customer segment
- **Customer Prediction**: Predict which segment a new customer belongs to
- **Marketing Recommendations**: Provides actionable marketing strategies for each customer segment

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required libraries: streamlit, pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/customer-segmentation-analysis.git
cd customer-segmentation-analysis
```
 2.Create a virtual environment (recommended):
```bash
bashpython:   -m venv venv

activate environment

source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

5. Open your browser and go to `http://localhost:8501`

## 📝 Usage Guide

### Upload & Analyze Data Mode

1. Select "Upload & Analyze Data" mode from the sidebar
2. Upload your customer data CSV file
3. Review the automatic feature detection results
4. Handle any missing values if present
5. Select features for clustering analysis
6. Choose the number of clusters (default: 5)
7. Click "Run Clustering Analysis" to generate insights

### Customer Prediction Mode

1. Select "Customer Prediction" mode from the sidebar
2. Enter customer information using the sliders
3. Click "Analyze Customer" to see:
   - The predicted customer segment
   - A description of the segment
   - Marketing recommendations
   - Visual representation of the customer's position relative to other segments

## 📊 Example Dataset Format

Your CSV file should contain customer attributes such as:
- Demographics (age, income, etc.)
- Purchase history
- Engagement metrics
- Behavioral data

Example columns:
```
customer_id, age, income, spending_score, purchase_frequency, last_purchase_days, loyalty_points
```
or any type of Customer dataset will work no it 
## 🔍 How It Works

1. **Data Preprocessing**: 
   - The app identifies numeric and categorical features
   - Handles missing values
   - Excludes ID columns from analysis

2. **Clustering**: 
   - Uses K-means algorithm to find natural groupings in the data
   - Standardizes features to give equal importance to all variables

3. **Visualization**: 
   - Creates insightful visualizations to understand customer segments
   - Offers both 2D and 3D visualization options

4. **Interpretation**: 
   - Generates descriptions based on the most distinctive features of each cluster
   - Provides statistical analysis of each segment

5. **Prediction**: 
   - Uses the trained model to predict which segment a new customer belongs to
   - Visualizes the customer's position relative to all segments

## 🛠️ Customization

- Adjust the number of clusters based on your business needs
- Select different feature combinations for analysis
- Experiment with different visualization methods

### 📫 Let's Connect!
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/om-shikhare)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ompythoncode@gmail.com,omjobshikhare@gmail.com)
Project Link:([https://github.com/yourusername/customer-segmentation-analysis](https://customer-segmentation-wztbyzzjgxyhzzoesc2vz6.streamlit.app/))

---

⭐️ If you found this project helpful, please give it a star on GitHub! ⭐️
