import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Set page configuration
st.set_page_config(
    page_title="Mall Customer Prediction",
    page_icon="🛍️",
    layout="wide"
)

# Title and description
st.title("Mall Customer Purchase Prediction")
st.markdown("This app predicts whether a customer is likely to make purchases at your mall based on their demographics.")

# Sidebar for user input
st.sidebar.header("Customer Information")

# Gender selection
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Age slider
age = st.sidebar.slider("Age", 18, 85, 30)

# Annual Income slider
annual_income = st.sidebar.slider("Annual Income (k$)", 15, 150, 50)

# Load the model and data
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        df = pd.read_csv("mall_customers.csv")
        return model, df
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None

# Load the model and dataset
model, df = load_model()

# Function to predict customer segment
def predict_segment(gender, age, annual_income):
    # Since KMeans doesn't use gender directly, we'll convert it to numerical
    # (though it's not used in the prediction directly)
    gender_num = 0 if gender == "Male" else 1
    
    # Create input data
    customer_data = np.array([[age, annual_income, 0]])  # Initial spending score set to 0
    
    # We need to find the cluster this customer belongs to based on age and income
    # For this example, let's use the 5-cluster model based on age, income, and spending score
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    
    # Find the closest cluster
    distances = []
    for center in kmeans.cluster_centers_:
        # Compare just age and income
        dist = np.sqrt((center[0] - age)**2 + (center[1] - annual_income)**2)
        distances.append(dist)
    
    cluster = np.argmin(distances)
    
    # Get the average spending score for this cluster
    cluster_data = df[kmeans.labels_ == cluster]
    avg_spending_score = cluster_data['Spending Score (1-100)'].mean()
    
    return cluster, avg_spending_score

# Main panel
if model is not None and df is not None:
    st.header("Customer Analysis")
    
    # Analyze button
    if st.button("Analyze Customer"):
        cluster, spending_score = predict_segment(gender, age, annual_income)
        
        # Display results
        st.subheader("Prediction Results")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Spending Score", f"{spending_score:.1f}/100")
            
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
            
            # Customer segment description
            segment_descriptions = {
                0: "Budget-conscious shoppers with moderate income, typically older",
                1: "High-income, moderate spenders who are selective in purchases",
                2: "Young enthusiastic shoppers with moderate income and high spending",
                3: "Low-income, careful shoppers who spend occasionally",
                4: "Target customers: High-income, high-spending premium shoppers"
            }
            
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
            # Visualize the customer position in the clustering
            st.subheader("Customer Positioning")
            
            # Create a scatter plot showing clusters and the new customer
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot existing clusters (only age and income dimensions)
            clusters = kmeans.fit_predict(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
            scatter = ax.scatter(
                df['Age'], 
                df['Annual Income (k$)'],
                c=clusters,
                alpha=0.6,
                cmap='viridis'
            )
            
            # Plot the new customer
            ax.scatter(age, annual_income, color='red', marker='*', s=300, label='New Customer')
            
            # Add labels and legend
            ax.set_xlabel('Age')
            ax.set_ylabel('Annual Income (k$)')
            ax.legend()
            
            # Display the plot
            st.pyplot(fig)
    
    # Show dataset overview
    st.header("Dataset Overview")
    st.dataframe(df.head())
    
    # Simple statistics about clusters
    st.header("Customer Segments Overview")
    
    # Perform clustering for visualization
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    
    # Show cluster statistics
    cluster_stats = df.groupby('Cluster').agg({
        'Age': 'mean',
        'Annual Income (k$)': 'mean',
        'Spending Score (1-100)': 'mean',
        'Gender': lambda x: (x == 'Female').mean() * 100  # Percentage of females
    }).reset_index()
    
    cluster_stats.columns = ['Cluster', 'Avg Age', 'Avg Income (k$)', 'Avg Spending Score', 'Female (%)']
    cluster_stats = cluster_stats.round(1)
    
    st.dataframe(cluster_stats)
    
    # Visualization of the clusters
    st.header("Cluster Visualization")
    
    viz_type = st.selectbox(
        "Select Visualization", 
        ["Age vs Spending Score", "Income vs Spending Score", "Age vs Income", "3D Visualization"]
    )
    
    fig = plt.figure(figsize=(10, 6))
    
    if viz_type == "Age vs Spending Score":
        plt.scatter(df['Age'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
        plt.xlabel('Age')
        plt.ylabel('Spending Score (1-100)')
        
    elif viz_type == "Income vs Spending Score":
        plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        
    elif viz_type == "Age vs Income":
        plt.scatter(df['Age'], df['Annual Income (k$)'], c=df['Cluster'], cmap='viridis')
        plt.xlabel('Age')
        plt.ylabel('Annual Income (k$)')
        
    else:  # 3D visualization
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            df['Age'], 
            df['Annual Income (k$)'], 
            df['Spending Score (1-100)'],
            c=df['Cluster'],
            cmap='viridis'
        )
        ax.set_xlabel('Age')
        ax.set_ylabel('Annual Income (k$)')
        ax.set_zlabel('Spending Score (1-100)')
    
    st.pyplot(fig)

else:
    st.error("Failed to load model or dataset. Please check your files.")

st.sidebar.markdown("---")
st.sidebar.info("""
    This app uses K-means clustering to segment customers based on their demographics and predict their shopping behavior.
    
    For best results, provide accurate customer information.
""")
