import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import io
import joblib
import os
import re

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="🛍️",
    layout="wide"
)

# Title and description
st.title("Customer Segmentation Analysis & Prediction")
st.markdown("""
    This app analyzes customer data and predicts customer segments based on their characteristics.
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

# Function to identify numeric and categorical columns
def identify_column_types(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove ID columns from feature lists
    id_cols = []
    for col in numeric_cols + categorical_cols:
        if re.search(r'id$|^id|customer|index|num', col.lower()):
            if col in numeric_cols:
                numeric_cols.remove(col)
            elif col in categorical_cols:
                categorical_cols.remove(col)
            id_cols.append(col)
    
    return numeric_cols, categorical_cols, id_cols

# Function to train model
def train_model(data, features, n_clusters=5):
    # Use selected features from the dataset
    X = data[features].values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    return kmeans, scaler

# Function to predict segment
def predict_segment(model, scaler, features_dict):
    # Convert features dict to array and reshape
    customer_data = np.array(list(features_dict.values())).reshape(1, -1)
    
    # Scale the data
    customer_data_scaled = scaler.transform(customer_data)
    
    # Predict cluster
    cluster = model.predict(customer_data_scaled)[0]
    
    return cluster

# Function to generate cluster descriptions
def generate_cluster_descriptions(data, cluster_column, feature_cols):
    descriptions = {}
    
    # Calculate mean values for each cluster
    cluster_means = data.groupby(cluster_column)[feature_cols].mean()
    
    # Generate descriptions based on relative positions
    for cluster in cluster_means.index:
        description = "Customers with "
        
        # Get top 2 distinguishing features (highest values)
        top_features = cluster_means.loc[cluster].nlargest(2)
        description += f"high {top_features.index[0].replace('_', ' ')} and {top_features.index[1].replace('_', ' ')}"
        
        # Get bottom feature (lowest value)
        bottom_feature = cluster_means.loc[cluster].nsmallest(1)
        description += f", but relatively low {bottom_feature.index[0].replace('_', ' ')}"
        
        descriptions[cluster] = description
    
    return descriptions

# Function to generate EDA plots
def generate_plots(data, numeric_cols, categorical_cols):
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution Plots", 
        "Categorical Analysis", 
        "Correlation Analysis", 
        "Clustering Results"
    ])
    
    with tab1:
        st.subheader("Distribution of Numeric Features")
        # Create subplots based on number of numeric columns
        num_plots = len(numeric_cols)
        rows = (num_plots // 3) + (1 if num_plots % 3 > 0 else 0)
        cols = min(3, num_plots)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])  # Make it indexable
        axes = axes.flatten()
        
        # Plot distributions
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                sns.histplot(data[col], bins=20, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
        
        # Hide unused subplots
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        if categorical_cols:
            st.subheader("Categorical Features Analysis")
            
            # Select categorical column to analyze
            cat_col = st.selectbox("Select categorical feature to analyze", categorical_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Category count
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(y=cat_col, data=data, ax=ax)
                ax.set_title(f'{cat_col} Distribution')
                st.pyplot(fig)
            
            with col2:
                # Category vs numeric feature
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_to_analyze = st.selectbox("Select numeric feature to analyze by category", numeric_cols)
                sns.violinplot(x=cat_col, y=feature_to_analyze, data=data, ax=ax)
                ax.set_title(f'{feature_to_analyze} by {cat_col}')
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.info("No categorical features found in the dataset.")
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        # Generate correlation matrix
        corr = data[numeric_cols].corr()
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)
        
        # Scatter plot of selected features
        st.subheader("Relationship between Features")
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Select X-axis feature", numeric_cols, index=0)
        with col2:
            y_feature = st.selectbox("Select Y-axis feature", numeric_cols, index=min(1, len(numeric_cols)-1))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=x_feature, y=y_feature, data=data, ax=ax)
        ax.set_title(f"Relationship between {x_feature} and {y_feature}")
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Customer Clustering Results")
        
        # Add cluster column if not present
        if "Cluster" not in data.columns:
            st.warning("Clustering not performed yet. Please run clustering first.")
            return
        
        # Select visualization features
        cluster_data = data.copy()
        
        if len(numeric_cols) >= 3:
            viz_features = st.multiselect(
                "Select features for visualization (max 3)", 
                numeric_cols,
                default=numeric_cols[:3]
            )
            
            if len(viz_features) < 2:
                st.warning("Please select at least 2 features for visualization")
                return
            
            # Choose visualization type
            if len(viz_features) == 2:
                viz_type = "2D"
            elif len(viz_features) >= 3:
                viz_type = st.selectbox("Select Visualization Type", ["2D", "3D"])
                viz_features = viz_features[:3]  # Limit to 3 features for 3D viz
            
            if viz_type == "2D":
                if len(viz_features) >= 2:
                    x_feature, y_feature = viz_features[:2]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(
                        cluster_data[x_feature], 
                        cluster_data[y_feature],
                        c=cluster_data['Cluster'], 
                        cmap='viridis',
                        s=50
                    )
                    ax.set_xlabel(x_feature)
                    ax.set_ylabel(y_feature)
                    ax.set_title(f'Customer Segments: {x_feature} vs {y_feature}')
                    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
                    st.pyplot(fig)
            else:  # 3D visualization
                if len(viz_features) >= 3:
                    x_feature, y_feature, z_feature = viz_features[:3]
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(
                        cluster_data[x_feature], 
                        cluster_data[y_feature], 
                        cluster_data[z_feature],
                        c=cluster_data['Cluster'],
                        cmap='viridis',
                        s=40
                    )
                    
                    ax.set_xlabel(x_feature)
                    ax.set_ylabel(y_feature)
                    ax.set_zlabel(z_feature)
                    ax.set_title('3D Visualization of Customer Segments')
                    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
                    
                    st.pyplot(fig)
        else:
            st.warning("Need at least 2 numeric features for clustering visualization")
            return
        
        # Show cluster statistics
        st.subheader("Cluster Statistics")
        
        # Numeric features by cluster
        cluster_stats = cluster_data.groupby('Cluster')[numeric_cols].mean().reset_index()
        
        # Add categorical features if present
        if categorical_cols:
            for cat_col in categorical_cols:
                # For each categorical column, get the most common value per cluster
                cat_stats = cluster_data.groupby('Cluster')[cat_col].agg(
                    lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else "Unknown"
                ).reset_index()
                cat_stats.columns = ['Cluster', f'Most Common {cat_col}']
                
                # Merge with main stats
                cluster_stats = pd.merge(cluster_stats, cat_stats, on='Cluster')
        
        st.dataframe(cluster_stats.round(2))
        
        # Cluster interpretations
        st.subheader("Cluster Interpretations")
        cluster_descriptions = generate_cluster_descriptions(cluster_data, 'Cluster', numeric_cols)
        
        for cluster, description in cluster_descriptions.items():
            st.write(f"**Cluster {cluster}:** {description}")

# Function to run clustering
def run_clustering(data, numeric_cols, n_clusters):
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[numeric_cols])
    
    # Create KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster to data
    data['Cluster'] = clusters
    
    return kmeans, scaler, data

# Function to generate marketing recommendations
def generate_recommendations(cluster, numeric_features):
    # Generic recommendations based on relative position
    recommendations = {
        "High value customers": "Focus on premium products and personalized services",
        "Medium value customers": "Targeted promotions and loyalty programs",
        "Budget-conscious customers": "Value-based offers and budget-friendly options",
        "Young enthusiasts": "Trendy products and social shopping experiences",
        "Selective spenders": "Quality-focused messaging and exclusive deals"
    }
    
    cluster_types = ["Budget-conscious customers", "Medium value customers", 
                     "High value customers", "Young enthusiasts", "Selective spenders"]
    
    # Return relevant recommendation
    return recommendations[cluster_types[cluster % len(cluster_types)]]

# Main application layout
st.sidebar.header("Choose Action")
app_mode = st.sidebar.selectbox("Select Mode", ["Upload & Analyze Data", "Customer Prediction"])

if app_mode == "Upload & Analyze Data":
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Load the data
        data = load_data(uploaded_file)
        
        if data is not None:
            # Show dataset overview
            st.header("Dataset Overview")
            st.write(f"Dataset Shape: {data.shape}")
            st.dataframe(data.head())
            
            # Identify column types
            numeric_cols, categorical_cols, id_cols = identify_column_types(data)
            
            # Check for missing values
            missing_values = data.isnull().sum()
            if missing_values.sum() > 0:
                st.warning("Dataset contains missing values:")
                st.write(missing_values[missing_values > 0])
                
                # Handle missing values
                handle_missing = st.radio("How to handle missing values?", 
                                          ["Drop rows with missing values", 
                                           "Fill numeric with mean, categorical with mode"])
                
                if handle_missing == "Drop rows with missing values":
                    data = data.dropna()
                    st.write(f"After dropping rows: {data.shape}")
                else:
                    # Fill numeric with mean
                    for col in numeric_cols:
                        if data[col].isnull().sum() > 0:
                            data[col] = data[col].fillna(data[col].mean())
                    
                    # Fill categorical with mode
                    for col in categorical_cols:
                        if data[col].isnull().sum() > 0:
                            data[col] = data[col].fillna(data[col].mode()[0])
                    
                    st.write("Missing values filled")
            
            # Display identified columns
            st.header("Feature Selection")
            st.write("The following columns were automatically identified:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Numeric Features:**")
                st.write(", ".join(numeric_cols))
            
            with col2:
                st.write("**Categorical Features:**")
                st.write(", ".join(categorical_cols) if categorical_cols else "None")
            
            with col3:
                st.write("**ID Columns (excluded):**")
                st.write(", ".join(id_cols) if id_cols else "None")
            
            # Feature selection
            st.subheader("Select Features for Clustering")
            selected_features = st.multiselect(
                "Choose features to use for clustering analysis",
                numeric_cols,
                default=numeric_cols
            )
            
            if not selected_features:
                st.warning("Please select at least one feature for clustering")
            else:
                # Get number of clusters
                n_clusters = st.slider("Number of clusters", 2, 10, 5)
                
                # Run clustering button
                if st.button("Run Clustering Analysis"):
                    with st.spinner("Running clustering..."):
                        # Train model with selected features
                        kmeans, scaler, clustered_data = run_clustering(data, selected_features, n_clusters)
                        
                        # Generate analysis plots
                        st.header("Data Analysis")
                        generate_plots(clustered_data, numeric_cols, categorical_cols)
                        
                        # Save the model and scaler for prediction
                        with open('model.pkl', 'wb') as f:
                            joblib.dump((kmeans, scaler, selected_features), f)
                        
                        # Also save column info
                        with open('column_info.pkl', 'wb') as f:
                            joblib.dump((numeric_cols, categorical_cols), f)
                        
                        st.success("Analysis complete and model trained! You can now use the Customer Prediction mode.")
    else:
        st.info("Please upload a CSV file to analyze.")

else:  # Customer Prediction mode
    # Check if model exists
    if not os.path.exists('model.pkl') or not os.path.exists('column_info.pkl'):
        st.warning("Please upload data and analyze it first to train the model.")
        st.info("Go to 'Upload & Analyze Data' mode to train the model with your data.")
    else:
        # Load the model, scaler and features
        with open('model.pkl', 'rb') as f:
            kmeans, scaler, selected_features = joblib.load(f)
        
        # Load column info
        with open('column_info.pkl', 'rb') as f:
            numeric_cols, categorical_cols = joblib.load(f)
        
        st.sidebar.header("Customer Information")
        
        # Input fields for features
        feature_values = {}
        
        for feature in selected_features:
            # Find min and max for reasonable slider ranges
            min_val = 0
            max_val = 100
            default_val = 50
            
            if feature.lower().find('age') >= 0:
                min_val, max_val, default_val = 18, 85, 30
            elif feature.lower().find('income') >= 0:
                min_val, max_val, default_val = 10, 200, 50
            elif feature.lower().find('score') >= 0 or feature.lower().find('rating') >= 0:
                min_val, max_val, default_val = 1, 100, 50
            
            # Create slider for each feature
            feature_values[feature] = st.sidebar.slider(
                f"{feature}", 
                min_val, max_val, default_val
            )
        
        # Analyze button
        if st.sidebar.button("Analyze Customer"):
            st.header("Customer Analysis")
            
            # Predict customer segment
            cluster = predict_segment(kmeans, scaler, feature_values)
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Display results
                st.subheader("Prediction Results")
                
                # Display customer details
                st.write("**Customer Details:**")
                for feature, value in feature_values.items():
                    st.write(f"- {feature}: {value}")
                
                # Customer segment info
                st.subheader(f"Customer Segment: Group {cluster+1}")
                
                # Generate cluster descriptions
                cluster_descriptions = {}
                for i in range(kmeans.n_clusters):
                    # Simple descriptions based on the cluster centers
                    centers = scaler.inverse_transform(kmeans.cluster_centers_)
                    center_df = pd.DataFrame(centers, columns=selected_features)
                    
                    # Get the defining characteristics
                    high_features = []
                    low_features = []
                    
                    for feature in selected_features:
                        # Compare this cluster's value with the average of all clusters
                        if center_df.loc[i, feature] > center_df[feature].mean():
                            high_features.append(feature)
                        else:
                            low_features.append(feature)
                    
                    # Create description
                    if high_features and low_features:
                        description = f"Customers with high {' and '.join(high_features[:2])}"
                        if low_features:
                            description += f", but lower {low_features[0]}"
                    else:
                        description = "General customers"
                    
                    cluster_descriptions[i] = description
                
                st.write(cluster_descriptions[cluster])
                
                # Marketing recommendations
                st.subheader("Marketing Recommendations")
                recommendation = generate_recommendations(cluster, selected_features)
                st.write(recommendation)
            
            with col2:
                # Create a 2D visualization showing the customer position
                if len(selected_features) >= 2:
                    st.subheader("Customer Positioning")
                    
                    # Select features for visualization
                    if len(selected_features) > 2:
                        viz_features = st.multiselect(
                            "Select two features for visualization", 
                            selected_features,
                            default=selected_features[:2]
                        )
                        if len(viz_features) != 2:
                            st.warning("Please select exactly 2 features")
                            viz_features = selected_features[:2]
                    else:
                        viz_features = selected_features
                    
                    if len(viz_features) == 2:
                        x_feature, y_feature = viz_features
                        
                        # Create mock data for better visualization
                        n_points = 200
                        mock_data = pd.DataFrame()
                        
                        # Generate realistic looking data based on cluster centers
                        centers = scaler.inverse_transform(kmeans.cluster_centers_)
                        for i in range(kmeans.n_clusters):
                            # Generate points around each center
                            n_cluster_points = n_points // kmeans.n_clusters
                            
                            cluster_data = {}
                            for j, feature in enumerate(selected_features):
                                # Add some noise around the center
                                center_val = centers[i, j]
                                std_dev = center_val * 0.2  # 20% variation
                                cluster_data[feature] = np.random.normal(center_val, std_dev, n_cluster_points)
                            
                            # Add cluster label
                            cluster_df = pd.DataFrame(cluster_data)
                            cluster_df['Cluster'] = i
                            
                            # Append to mock data
                            mock_data = pd.concat([mock_data, cluster_df], ignore_index=True)
                        
                        # Create the plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot mock data
                        ax.scatter(
                            mock_data[x_feature], 
                            mock_data[y_feature],
                            c=mock_data['Cluster'],
                            alpha=0.5,
                            cmap='viridis'
                        )
                        
                        # Plot the customer data
                        ax.scatter(
                            feature_values[x_feature], 
                            feature_values[y_feature], 
                            color='red', marker='*', s=300, label='New Customer'
                        )
                        
                        ax.set_xlabel(x_feature)
                        ax.set_ylabel(y_feature)
                        ax.set_title('Customer Positioning')
                        ax.legend()
                        
                        st.pyplot(fig)
                else:
                    st.warning("Need at least 2 numeric features for visualization")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
    This app uses K-means clustering to segment customers based on their characteristics and predict their behavior.
    
    For best results, provide accurate customer information and select relevant features for analysis.
""")
