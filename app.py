import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import io
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Mall Customer Analysis & Prediction",
    page_icon="🛍️",
    layout="wide"
)

# Title and description
st.title("Mall Customer Analysis & Prediction")
st.markdown("""
    This app analyzes mall customer data and predicts customer segments based on demographics.
    Upload your dataset or use the built-in prediction tool for individual customers.
""")

# Function to load dataset
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to train model
def train_model(data):
    # Use the features from the dataset
    X = data.select_dtypes(include=['int64', 'float64']).values
    
    # Create KMeans model with 5 clusters
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X)
    
    return kmeans

# Function to predict segment
def predict_segment(model, data, age, income, spending_score=None):
    # If spending score is not provided, we just use age and income
    if spending_score is None:
        customer_data = np.array([[age, income]])
        
        # Find the closest cluster
        distances = []
        for center in model.cluster_centers_:
            # Compare just age and income
            dist = np.sqrt((center[0] - age)**2 + (center[1] - income)**2)
            distances.append(dist)
        
        cluster = np.argmin(distances)
    else:
        # If all features are provided
        customer_data = np.array([[age, income, spending_score]])
        cluster = model.predict(customer_data)[0]
    
    return cluster

# Function to generate EDA plots
def generate_plots(data):
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribution Plots", 
        "Gender Analysis", 
        "Age Groups", 
        "Income vs Spending", 
        "Clustering"
    ])
    
    with tab1:
        st.subheader("Distribution of Features")
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # Plot distributions
        for i, col in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
            sns.histplot(data[col], bins=20, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
        
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Gender Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender count
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(y='Gender', data=data, ax=ax)
            ax.set_title('Gender Distribution')
            st.pyplot(fig)
        
        with col2:
            # Gender vs features
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_to_analyze = st.selectbox("Select feature to analyze by gender", 
                                             ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
            sns.violinplot(x='Gender', y=feature_to_analyze, data=data, ax=ax)
            ax.set_title(f'{feature_to_analyze} by Gender')
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Age Group Analysis")
        
        # Define age groups
        age_groups = {
            "18-25": data.Age[(data.Age >= 18) & (data.Age <= 25)],
            "26-35": data.Age[(data.Age >= 26) & (data.Age <= 35)],
            "36-45": data.Age[(data.Age >= 36) & (data.Age <= 45)],
            "46-55": data.Age[(data.Age >= 46) & (data.Age <= 55)],
            "55+": data.Age[data.Age >= 56]
        }
        
        agex = list(age_groups.keys())
        agey = [len(group.values) for group in age_groups.values()]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=agex, y=agey, palette="mako", ax=ax)
        ax.set_title("Number of Customers in Different Age Groups")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Income vs Spending Analysis")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=data, ax=ax)
        ax.set_title("Relationship between Income and Spending")
        st.pyplot(fig)
        
        # Income distribution
        income_groups = {
            "$ 0 - 30,000": len(data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 0) & (data["Annual Income (k$)"] <= 30)]),
            "$ 30,001 - 60,000": len(data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 31) & (data["Annual Income (k$)"] <= 60)]),
            "$ 60,001 - 90,000": len(data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 61) & (data["Annual Income (k$)"] <= 90)]),
            "$ 90,001 - 120,000": len(data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 91) & (data["Annual Income (k$)"] <= 120)]),
            "$ 120,001 - 150,000": len(data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 121) & (data["Annual Income (k$)"] <= 150)])
        }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=list(income_groups.keys()), y=list(income_groups.values()), palette="Spectral", ax=ax)
        ax.set_title("Annual Income Distribution")
        ax.set_xlabel("Income Range")
        ax.set_ylabel("Number of Customers")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with tab5:
        st.subheader("Customer Clustering")
        
        # Create cluster features
        X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
        
        # Create KMeans model with 5 clusters
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Add cluster to data
        cluster_data = data.copy()
        cluster_data['Cluster'] = clusters
        
        # Choose visualization type
        viz_type = st.selectbox(
            "Select Visualization", 
            ["Age vs Spending Score", "Income vs Spending Score", "Age vs Income", "3D Visualization"]
        )
        
        if viz_type == "Age vs Spending Score":
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                cluster_data['Age'], 
                cluster_data['Spending Score (1-100)'],
                c=cluster_data['Cluster'], 
                cmap='viridis',
                s=50
            )
            ax.set_xlabel('Age')
            ax.set_ylabel('Spending Score (1-100)')
            ax.set_title('Customer Segments: Age vs Spending Score')
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            st.pyplot(fig)
            
        elif viz_type == "Income vs Spending Score":
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                cluster_data['Annual Income (k$)'], 
                cluster_data['Spending Score (1-100)'],
                c=cluster_data['Cluster'], 
                cmap='viridis',
                s=50
            )
            ax.set_xlabel('Annual Income (k$)')
            ax.set_ylabel('Spending Score (1-100)')
            ax.set_title('Customer Segments: Income vs Spending Score')
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            st.pyplot(fig)
            
        elif viz_type == "Age vs Income":
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                cluster_data['Age'], 
                cluster_data['Annual Income (k$)'],
                c=cluster_data['Cluster'], 
                cmap='viridis',
                s=50
            )
            ax.set_xlabel('Age')
            ax.set_ylabel('Annual Income (k$)')
            ax.set_title('Customer Segments: Age vs Income')
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            st.pyplot(fig)
            
        else:  # 3D visualization
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                cluster_data['Age'], 
                cluster_data['Annual Income (k$)'], 
                cluster_data['Spending Score (1-100)'],
                c=cluster_data['Cluster'],
                cmap='viridis',
                s=40
            )
            
            ax.set_xlabel('Age')
            ax.set_ylabel('Annual Income (k$)')
            ax.set_zlabel('Spending Score (1-100)')
            ax.set_title('3D Visualization of Customer Segments')
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            
            st.pyplot(fig)
        
        # Show cluster statistics
        st.subheader("Cluster Statistics")
        cluster_stats = cluster_data.groupby('Cluster').agg({
            'Age': 'mean',
            'Annual Income (k$)': 'mean',
            'Spending Score (1-100)': 'mean',
            'Gender': lambda x: (x == 'Female').mean() * 100  # Percentage of females
        }).reset_index()
        
        cluster_stats.columns = ['Cluster', 'Avg Age', 'Avg Income (k$)', 'Avg Spending Score', 'Female (%)']
        cluster_stats = cluster_stats.round(1)
        
        st.dataframe(cluster_stats)
        
        # Cluster interpretations
        st.subheader("Cluster Interpretations")
        interpretations = {
            0: "Budget-conscious shoppers with moderate income, typically older",
            1: "High-income, moderate spenders who are selective in purchases",
            2: "Young enthusiastic shoppers with moderate income and high spending",
            3: "Low-income, careful shoppers who spend occasionally",
            4: "Target customers: High-income, high-spending premium shoppers"
        }
        
        for cluster, interpretation in interpretations.items():
            st.write(f"**Cluster {cluster}:** {interpretation}")

# Main application layout
st.sidebar.header("Choose Action")
app_mode = st.sidebar.selectbox("Select Mode", ["Upload & Analyze Data", "Customer Prediction"])

if app_mode == "Upload & Analyze Data":
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload Mall Customer Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Load the data
        data = load_data(uploaded_file)
        
        if data is not None:
            # Check if data has expected columns
            expected_columns = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            missing_columns = [col for col in expected_columns if col not in data.columns]
            
            if missing_columns:
                st.error(f"The uploaded data is missing required columns: {', '.join(missing_columns)}")
            else:
                # Show dataset overview
                st.header("Dataset Overview")
                st.write(f"Dataset Shape: {data.shape}")
                st.dataframe(data.head())
                
                # Check for missing values
                if data.isnull().sum().sum() > 0:
                    st.warning("Dataset contains missing values:")
                    st.write(data.isnull().sum())
                
                # Drop CustomerID if present
                if 'CustomerID' in data.columns:
                    data = data.drop(['CustomerID'], axis=1)
                
                # Generate analysis plots
                st.header("Data Analysis")
                generate_plots(data)
                
                # Train model and save
                model = train_model(data)
                
                # Save the model for prediction
                with open('model.pkl', 'wb') as f:
                    joblib.dump(model, f)
                
                st.success("Analysis complete and model trained! You can now use the Customer Prediction mode.")
    else:
        st.info("Please upload a CSV file to analyze.")

else:  # Customer Prediction mode
    st.sidebar.header("Customer Information")
    
    # Gender selection
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    
    # Age slider
    age = st.sidebar.slider("Age", 18, 85, 30)
    
    # Annual Income slider
    annual_income = st.sidebar.slider("Annual Income (k$)", 15, 150, 50)
    
    # Optional: Spending Score
    include_spending = st.sidebar.checkbox("Include Spending Score?")
    spending_score = None
    if include_spending:
        spending_score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)
    
    # Check if model exists
    if os.path.exists('model.pkl'):
        # Load the model
        model = joblib.load('model.pkl')
        
        # Analyze button
        if st.sidebar.button("Analyze Customer"):
            st.header("Customer Analysis")
            
            # Predict customer segment
            cluster = predict_segment(model, None, age, annual_income, spending_score)
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Display results
                st.subheader("Prediction Results")
                
                # Customer segment description
                segment_descriptions = {
                    0: "Budget-conscious shoppers with moderate income, typically older",
                    1: "High-income, moderate spenders who are selective in purchases",
                    2: "Young enthusiastic shoppers with moderate income and high spending",
                    3: "Low-income, careful shoppers who spend occasionally",
                    4: "Target customers: High-income, high-spending premium shoppers"
                }
                
                # Estimated spending score if not provided
                if spending_score is None:
                    # Estimate spending from cluster averages
                    estimated_scores = {
                        0: 40,  # Budget-conscious
                        1: 55,  # High-income moderate
                        2: 85,  # Young enthusiastic
                        3: 20,  # Low-income careful
                        4: 90   # Premium shoppers
                    }
                    spending_score = estimated_scores[cluster]
                    st.metric("Estimated Spending Score", f"{spending_score}/100")
                else:
                    st.metric("Spending Score", f"{spending_score}/100")
                
                # Purchase likelihood interpretation
                if spending_score >= 70:
                    purchase_likelihood = "High"
                    color = "green"
                elif spending_score >= 40:
                    purchase_likelihood = "Medium"
                    color = "orange"
                else:
                    purchase_likelihood = "Low"
                    color = "red"
                
                st.markdown(f"<h3 style='color:{color}'>Purchase Likelihood: {purchase_likelihood}</h3>", unsafe_allow_html=True)
                
                st.subheader(f"Customer Segment: Group {cluster+1}")
                st.write(segment_descriptions[cluster])
                
                # Marketing recommendations
                st.subheader("Marketing Recommendations")
                recommendations = {
                    0: "Offer value-based promotions and loyalty programs",
                    1: "Focus on quality and exclusive products with targeted discounts",
                    2: "Promote trendy items and social shopping experiences",
                    3: "Provide budget-friendly options and special sales",
                    4: "Showcase premium products and personalized shopping experiences"
                }
                st.write(recommendations[cluster])
            
            with col2:
                # Create a 2D visualization showing the customer position
                st.subheader("Customer Positioning")
                
                # Create a sample dataset for visualization
                sample_data = pd.DataFrame({
                    'Age': [age],
                    'Annual Income (k$)': [annual_income],
                    'Spending Score (1-100)': [spending_score],
                    'Cluster': [cluster]
                })
                
                # Choose visualization type
                viz_type = st.selectbox(
                    "Select Visualization", 
                    ["Age vs Spending Score", "Income vs Spending Score", "Age vs Income"]
                )
                
                # Create mock data for better visualization
                mock_data = pd.DataFrame({
                    'Age': np.random.randint(18, 85, 200),
                    'Annual Income (k$)': np.random.randint(15, 150, 200),
                    'Spending Score (1-100)': np.random.randint(1, 100, 200)
                })
                
                # Add clusters to mock data
                X_mock = mock_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
                mock_data['Cluster'] = model.predict(X_mock)
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if viz_type == "Age vs Spending Score":
                    # Plot mock data
                    ax.scatter(
                        mock_data['Age'], 
                        mock_data['Spending Score (1-100)'],
                        c=mock_data['Cluster'],
                        alpha=0.5,
                        cmap='viridis'
                    )
                    
                    # Plot the customer data
                    ax.scatter(age, spending_score, color='red', marker='*', s=300, label='New Customer')
                    ax.set_xlabel('Age')
                    ax.set_ylabel('Spending Score (1-100)')
                    
                elif viz_type == "Income vs Spending Score":
                    # Plot mock data
                    ax.scatter(
                        mock_data['Annual Income (k$)'], 
                        mock_data['Spending Score (1-100)'],
                        c=mock_data['Cluster'],
                        alpha=0.5,
                        cmap='viridis'
                    )
                    
                    # Plot the customer data
                    ax.scatter(annual_income, spending_score, color='red', marker='*', s=300, label='New Customer')
                    ax.set_xlabel('Annual Income (k$)')
                    ax.set_ylabel('Spending Score (1-100)')
                    
                else:  # Age vs Income
                    # Plot mock data
                    ax.scatter(
                        mock_data['Age'], 
                        mock_data['Annual Income (k$)'],
                        c=mock_data['Cluster'],
                        alpha=0.5,
                        cmap='viridis'
                    )
                    
                    # Plot the customer data
                    ax.scatter(age, annual_income, color='red', marker='*', s=300, label='New Customer')
                    ax.set_xlabel('Age')
                    ax.set_ylabel('Annual Income (k$)')
                
                ax.legend()
                st.pyplot(fig)
    else:
        st.warning("Please upload data and analyze it first to train the model.")
        st.info("Go to 'Upload & Analyze Data' mode to train the model with your data.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
    This app uses K-means clustering to segment customers based on their demographics and predict their shopping behavior.
    
    For best results, provide accurate customer information.
""")
