import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation Tool",
    page_icon="📊",
    layout="wide"
)

# Add CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #4B5563;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Create header
st.markdown("<h1 class='main-header'>Customer Segmentation Analysis</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h2 class='sub-header'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Choose a page:", ["Data Overview", "Cluster Analysis", "Prediction"])

@st.cache_data
def load_data():
    """Load and cache data from CSV file"""
    try:
        df = pd.read_csv("mall_customers.csv")
        df.drop(["CustomerID"], axis=1, inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: mall_customers.csv file not found!")
        return None

# Helper functions for plotting
def plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)

def plot_income_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Annual Income (k$)"], bins=20, kde=True, ax=ax)
    ax.set_title("Annual Income Distribution")
    st.pyplot(fig)

def plot_spending_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Spending Score (1-100)"], bins=20, kde=True, ax=ax)
    ax.set_title("Spending Score Distribution")
    st.pyplot(fig)

def plot_gender_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x="Gender", data=df, ax=ax)
    ax.set_title("Gender Distribution")
    st.pyplot(fig)

def create_plotly_3d_clusters(df, model):
    # Add cluster labels to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = model.labels_
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df_with_clusters, 
        x='Age', 
        y='Annual Income (k$)', 
        z='Spending Score (1-100)',
        color='Cluster', 
        symbol='Gender',
        opacity=0.7,
        title='Customer Segments in 3D Space'
    )
    
    # Add centroids
    centroids = model.cluster_centers_
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers',
        marker=dict(color='black', size=10, symbol='diamond'),
        name='Centroids'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Age',
            yaxis_title='Annual Income (k$)',
            zaxis_title='Spending Score (1-100)'
        ),
        legend_title_text='Customer Segments'
    )
    
    return fig

def get_cluster_description(cluster_id, centroids):
    """Generate a description for each cluster based on centroid values"""
    descriptions = {
        0: "Young customers with low income and medium spending score",
        1: "Middle-aged customers with medium income and high spending score",
        2: "Young customers with high income and high spending score",
        3: "Senior customers with medium income and low spending score",
        4: "Middle-aged customers with high income and low spending score"
    }
    return descriptions.get(cluster_id, "Cluster information not available")

def train_kmeans_model(df, n_clusters=5):
    """Train a KMeans model with the given number of clusters"""
    # Convert categorical to numerical
    df_encoded = df.copy()
    if 'Gender' in df.columns:
        df_encoded['Gender'] = df_encoded['Gender'].map({'Female': 0, 'Male': 1})
        
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_encoded)
    
    # Train KMeans
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    
    return kmeans, scaler

def load_or_train_model(df):
    """Load a pre-trained model or train a new one"""
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.info("Training new model...")
        model, scaler = train_kmeans_model(df)
        # Save the model
        joblib.dump(model, 'kmeans_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        return model, scaler

# Load data
df = load_data()

if df is not None:
    # Data Overview Page
    if page == "Data Overview":
        st.markdown("<h2 class='sub-header'>Data Overview</h2>", unsafe_allow_html=True)
        
        # Display basic information
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Dataset Shape")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        with col2:
            st.write("### Missing Values")
            st.write(f"Total missing values: {df.isnull().sum().sum()}")
        
        # Display the dataframe
        st.write("### Sample Data")
        st.dataframe(df.head(10))
        
        # Display statistics
        st.write("### Statistical Summary")
        st.dataframe(df.describe())
        
        # Visualizations
        st.markdown("<h2 class='sub-header'>Data Visualizations</h2>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Age", "Income", "Spending Score", "Gender"])
        
        with tab1:
            plot_age_distribution(df)
            
        with tab2:
            plot_income_distribution(df)
            
        with tab3:
            plot_spending_distribution(df)
            
        with tab4:
            plot_gender_distribution(df)
        
        # Correlation heatmap
        st.write("### Feature Correlation")
        df_encoded = df.copy()
        if 'Gender' in df.columns:
            df_encoded['Gender'] = df_encoded['Gender'].map({'Female': 0, 'Male': 1})
            
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
    # Cluster Analysis Page
    elif page == "Cluster Analysis":
        st.markdown("<h2 class='sub-header'>Cluster Analysis</h2>", unsafe_allow_html=True)
        
        # Train or load model
        with st.spinner("Loading model..."):
            model, scaler = load_or_train_model(df)
        
        st.success("Model loaded successfully!")
        
        # Display number of clusters
        st.write(f"### Number of Clusters: {model.n_clusters}")
        
        # Display centroids
        st.write("### Cluster Centroids")
        centroids = model.cluster_centers_
        centroid_df = pd.DataFrame(centroids, columns=df.columns)
        st.dataframe(centroid_df)
        
        # 3D visualization
        st.write("### 3D Visualization of Clusters")
        with st.spinner("Generating 3D visualization..."):
            fig = create_plotly_3d_clusters(df, model)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster descriptions
        st.write("### Cluster Descriptions")
        for i in range(model.n_clusters):
            st.markdown(f"**Cluster {i}**: {get_cluster_description(i, centroids)}")
        
        # Distribution within clusters
        st.write("### Cluster Distribution")
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = model.labels_
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Cluster', data=df_with_clusters, ax=ax)
        ax.set_title("Number of Customers in Each Cluster")
        st.pyplot(fig)
        
    # Prediction Page
    elif page == "Prediction":
        st.markdown("<h2 class='sub-header'>Customer Segmentation Prediction</h2>", unsafe_allow_html=True)
        
        st.markdown("<div class='highlight'>Enter customer information to predict their segment</div>", unsafe_allow_html=True)
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.radio("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 80, 30)
        
        with col2:
            income = st.slider("Annual Income (k$)", 10, 150, 50)
            spending = st.slider("Spending Score (1-100)", 1, 100, 50)
        
        # Create a dataframe for the input
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Annual Income (k$)': [income],
            'Spending Score (1-100)': [spending]
        })
        
        st.write("### Customer Information")
        st.dataframe(input_data)
        
        # Make prediction
        if st.button("Predict Segment"):
            with st.spinner("Making prediction..."):
                # Load model if not already loaded
                model, scaler = load_or_train_model(df)
                
                # Preprocess input data
                input_encoded = input_data.copy()
                input_encoded['Gender'] = input_encoded['Gender'].map({'Female': 0, 'Male': 1})
                
                # Scale the input
                input_scaled = scaler.transform(input_encoded)
                
                # Predict the cluster
                cluster = model.predict(input_scaled)[0]
                
                # Display result
                st.success(f"Prediction complete!")
                st.markdown(f"<h3>Customer belongs to Segment {cluster}</h3>", unsafe_allow_html=True)
                st.markdown(f"<div class='highlight'>{get_cluster_description(cluster, model.cluster_centers_)}</div>", unsafe_allow_html=True)
                
                # Show where this customer falls in the overall distribution
                st.write("### Customer Position in Segments")
                
                # Add this new customer to a temporary dataframe for visualization
                df_temp = df.copy()
                df_temp['Cluster'] = model.labels_
                
                # Create a new row for the current customer with their predicted cluster
                new_customer = input_data.copy()
                new_customer['Cluster'] = cluster
                new_customer['Customer'] = 'New Customer'
                
                # Add Customer column to original data
                df_temp['Customer'] = 'Existing Customers'
                
                # Combine the dataframes
                plot_df = pd.concat([df_temp, new_customer])
                
                # Create a 3D scatter plot showing where the new customer falls
                fig = px.scatter_3d(
                    plot_df,
                    x='Age',
                    y='Annual Income (k$)', 
                    z='Spending Score (1-100)',
                    color='Cluster',
                    symbol='Customer',
                    opacity=0.7,
                    title='New Customer Position in Segments',
                    size_max=10,
                    size=[5 if c == 'Existing Customers' else 15 for c in plot_df['Customer']]
                )
                
                # Highlight the new customer
                fig.update_traces(
                    marker=dict(size=15, opacity=1.0),
                    selector=dict(mode='markers', name='New Customer')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Marketing recommendations based on cluster
                st.markdown("<h3 class='sub-header'>Marketing Recommendations</h3>", unsafe_allow_html=True)
                
                recommendations = {
                    0: [
                        "Target with budget-friendly products and introductory offers",
                        "Develop loyalty programs with gradual rewards",
                        "Offer educational content about product value"
                    ],
                    1: [
                        "Focus on premium products with good value proposition",
                        "Create exclusive membership benefits",
                        "Implement referral programs as they're likely to recommend"
                    ],
                    2: [
                        "Highlight luxury and trendy items",
                        "Create VIP experiences and early access to new products",
                        "Develop targeted social media campaigns for this tech-savvy segment"
                    ],
                    3: [
                        "Offer senior discounts and loyalty rewards",
                        "Focus on product durability and practicality",
                        "Provide excellent customer service with personal touch"
                    ],
                    4: [
                        "Emphasize high-quality, practical products",
                        "Develop rational marketing messages focusing on product benefits",
                        "Create time-saving shopping experiences"
                    ]
                }
                
                for i, rec in enumerate(recommendations.get(cluster, [])):
                    st.markdown(f"- {rec}")
else:
    st.error("Failed to load data. Please check if the mall_customers.csv file is in the correct location.")

# Add footer
st.markdown("---")
st.markdown("<p style='text-align: center'>Customer Segmentation Analysis Tool | Created with Streamlit</p>", unsafe_allow_html=True)