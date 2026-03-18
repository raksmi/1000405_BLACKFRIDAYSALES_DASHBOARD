import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy import stats

from mlxtend.frequent_patterns import apriori, association_rules

# Page configuration
st.set_page_config(page_title="Black Friday Sales Data Mining Dashboard", page_icon="🛍", layout="wide")

st.title("🛍 Black Friday Sales Data Mining Dashboard")
st.markdown("---")

# =============================================
# STAGE 1: PROJECT SCOPE DEFINITION
# =============================================
st.sidebar.header("📋 Project Overview")
st.sidebar.info("""
**Objectives:**
- Identify shopping behaviors
- Group customers into clusters
- Find product combinations bought together
- Detect unusual big spenders
""")

# Upload dataset
uploaded_file = st.file_uploader("Upload Black Friday Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Store original data for comparison
    df_original = df.copy()

    # =============================================
    # STAGE 1: PROJECT SCOPE (Display)
    # =============================================
    st.header("Stage 1: Project Scope Definition")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique Users", f"{df['User_ID'].nunique():,}")
    with col3:
        st.metric("Unique Products", f"{df['Product_ID'].nunique():,}")
    
    st.subheader("Dataset Columns")
    st.write("**Available columns:** " + ", ".join(df.columns.tolist()))
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    # =============================================
    # STAGE 2: DATA CLEANING & PREPROCESSING
    # =============================================
    st.header("Stage 2: Data Cleaning & Preprocessing")
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    missing_before = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_before.index,
        'Missing Values': missing_before.values,
        'Percentage': (missing_before.values / len(df) * 100).round(2)
    })
    st.dataframe(missing_df[missing_df['Missing Values'] > 0])
    
    # Handle missing values
    df['Product_Category_2'].fillna(0, inplace=True)
    df['Product_Category_3'].fillna(0, inplace=True)
    
    st.success("✅ Missing values in Product_Category_2 and Product_Category_3 filled with 0")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    st.subheader("Duplicate Check")
    st.write(f"Duplicate rows found: **{duplicates}**")
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        st.success(f"✅ Removed {duplicates} duplicate rows")
    
    # Encode categorical data
    st.subheader("Categorical Encoding")
    
    # Gender encoding (Male = 0, Female = 1)
    df['Gender_Encoded'] = df['Gender'].map({'M': 0, 'F': 1})
    st.write("• Gender encoded: M → 0, F → 1")
    
    # Age encoding with ordered mapping
    age_mapping = {
        '0-17': 1,
        '18-25': 2,
        '26-35': 3,
        '36-45': 4,
        '46-50': 5,
        '51-55': 6,
        '55+': 7
    }
    df['Age_Encoded'] = df['Age'].map(age_mapping)
    st.write("• Age groups encoded with ordered mapping (0-17 → 1, 18-25 → 2, etc.)")
    
    # City Category encoding
    le = LabelEncoder()
    df['City_Category_Encoded'] = le.fit_transform(df['City_Category'])
    st.write("• City Category encoded using LabelEncoder")
    
    # Stay in current city encoding
    stay_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4}
    df['Stay_Encoded'] = df['Stay_In_Current_City_Years'].map(stay_mapping)
    st.write("• Stay in Current City Years encoded (0-4)")
    
    # Normalize Purchase amounts
    st.subheader("Purchase Normalization")
    scaler_norm = MinMaxScaler()
    df['Purchase_Normalized'] = scaler_norm.fit_transform(df[['Purchase']])
    st.write("• Purchase amounts normalized using MinMaxScaler")
    
    # Show cleaned data
    st.subheader("Cleaned Dataset Preview")
    st.dataframe(df.head())

    # =============================================
    # STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
    # =============================================
    st.header("Stage 3: Exploratory Data Analysis (EDA)")
    
    # Purchase Distribution
    st.subheader("Purchase Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(df['Purchase'], bins=30, kde=True, ax=ax1, color='steelblue')
        ax1.set_title('Purchase Amount Distribution')
        ax1.set_xlabel('Purchase Amount')
        st.pyplot(fig1)
    with col2:
        fig1b, ax1b = plt.subplots()
        sns.boxplot(y=df['Purchase'], ax=ax1b, color='steelblue')
        ax1b.set_title('Purchase Amount Box Plot')
        st.pyplot(fig1b)
    
    # Purchase by Gender
    st.subheader("Purchase Analysis by Gender")
    col1, col2 = st.columns(2)
    with col1:
        fig2, ax2 = plt.subplots()
        gender_labels = {0: 'Male', 1: 'Female'}
        sns.barplot(x='Gender_Encoded', y='Purchase', data=df, ax=ax2, palette='Set2')
        ax2.set_xticklabels(['Male', 'Female'])
        ax2.set_title('Average Purchase by Gender')
        ax2.set_xlabel('Gender')
        st.pyplot(fig2)
    with col2:
        fig2b, ax2b = plt.subplots()
        sns.boxplot(x='Gender_Encoded', y='Purchase', data=df, ax=ax2b, palette='Set2')
        ax2b.set_xticklabels(['Male', 'Female'])
        ax2b.set_title('Purchase Distribution by Gender')
        ax2b.set_xlabel('Gender')
        st.pyplot(fig2b)
    
    # Purchase by Age Group
    st.subheader("Purchase Analysis by Age Group")
    col1, col2 = st.columns(2)
    with col1:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='Age', y='Purchase', data=df, ax=ax3, palette='viridis')
        ax3.set_title('Purchase Distribution by Age Group')
        ax3.set_xlabel('Age Group')
        ax3.tick_params(axis='x', rotation=45)
        st.pyplot(fig3)
    with col2:
        fig3b, ax3b = plt.subplots(figsize=(10, 5))
        age_purchase = df.groupby('Age')['Purchase'].mean().sort_values(ascending=False)
        sns.barplot(x=age_purchase.index, y=age_purchase.values, ax=ax3b, palette='viridis')
        ax3b.set_title('Average Purchase by Age Group')
        ax3b.set_xlabel('Age Group')
        ax3b.set_ylabel('Average Purchase')
        ax3b.tick_params(axis='x', rotation=45)
        st.pyplot(fig3b)
    
    # Most Popular Product Categories
    st.subheader("Product Category Popularity")
    col1, col2 = st.columns(2)
    with col1:
        fig_cat, ax_cat = plt.subplots(figsize=(10, 5))
        cat_counts = df['Product_Category_1'].value_counts().head(10)
        sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=ax_cat, palette='coolwarm')
        ax_cat.set_title('Top 10 Product Categories (Category 1)')
        ax_cat.set_xlabel('Product Category')
        ax_cat.set_ylabel('Number of Purchases')
        st.pyplot(fig_cat)
    with col2:
        fig_cat2, ax_cat2 = plt.subplots(figsize=(10, 5))
        cat_avg = df.groupby('Product_Category_1')['Purchase'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=cat_avg.index, y=cat_avg.values, ax=ax_cat2, palette='coolwarm')
        ax_cat2.set_title('Top 10 Categories by Avg Purchase Amount')
        ax_cat2.set_xlabel('Product Category')
        ax_cat2.set_ylabel('Average Purchase Amount')
        st.pyplot(fig_cat2)
    
    # Scatter plots for Purchase vs. Occupation and Stay
    st.subheader("Purchase vs. Occupation & Stay Duration")
    col1, col2 = st.columns(2)
    with col1:
        fig_scat1, ax_scat1 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(x='Occupation', y='Purchase', data=df.sample(min(1000, len(df))), 
                        alpha=0.5, ax=ax_scat1, hue='Gender', palette='Set1')
        ax_scat1.set_title('Purchase vs. Occupation (Sample)')
        st.pyplot(fig_scat1)
    with col2:
        fig_scat2, ax_scat2 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='Stay_In_Current_City_Years', y='Purchase', data=df, ax=ax_scat2, palette='Set3')
        ax_scat2.set_title('Purchase by Stay Duration in City')
        st.pyplot(fig_scat2)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = ['Gender_Encoded', 'Age_Encoded', 'Occupation', 'City_Category_Encoded', 
                    'Stay_Encoded', 'Marital_Status', 'Product_Category_1', 
                    'Product_Category_2', 'Product_Category_3', 'Purchase']
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax4, fmt='.2f')
    ax4.set_title('Feature Correlation Heatmap')
    st.pyplot(fig4)
    
    # City Category Analysis
    st.subheader("City Category Analysis")
    fig_city, ax_city = plt.subplots(figsize=(10, 5))
    city_stats = df.groupby('City_Category').agg({
        'Purchase': ['mean', 'count', 'sum']
    }).round(2)
    city_stats.columns = ['Avg Purchase', 'Total Transactions', 'Total Revenue']
    st.dataframe(city_stats)
    
    fig_city_bar, ax_city_bar = plt.subplots()
    sns.barplot(x=city_stats.index, y='Avg Purchase', data=city_stats, ax=ax_city_bar, palette='Set2')
    ax_city_bar.set_title('Average Purchase by City Category')
    st.pyplot(fig_city_bar)

    # =============================================
    # STAGE 4: CLUSTERING ANALYSIS
    # =============================================
    st.header("Stage 4: Clustering Analysis")
    
    st.subheader("Feature Selection for Clustering")
    cluster_features = st.multiselect(
        "Select features for clustering:",
        ['Age_Encoded', 'Occupation', 'Marital_Status', 'Purchase', 'Gender_Encoded', 
         'City_Category_Encoded', 'Stay_Encoded'],
        default=['Age_Encoded', 'Occupation', 'Marital_Status', 'Purchase']
    )
    
    if len(cluster_features) >= 2:
        features = df[cluster_features].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Elbow Method
        st.subheader("Elbow Method for Optimal K")
        inertias = []
        K_range = range(1, 11)
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(scaled_features)
            inertias.append(kmeans_temp.inertia_)
        
        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(K_range, inertias, 'bx-')
        ax_elbow.set_xlabel('Number of Clusters (K)')
        ax_elbow.set_ylabel('Inertia')
        ax_elbow.set_title('Elbow Method for Optimal K')
        st.pyplot(fig_elbow)
        
        # Select number of clusters
        n_clusters = st.slider("Select Number of Clusters:", min_value=2, max_value=10, value=4)
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(scaled_features)
        
        # Cluster visualization
        st.subheader("Cluster Visualization")
        
        # 2D Scatter plot
        col1, col2 = st.columns(2)
        with col1:
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            scatter = sns.scatterplot(x=features.iloc[:, 0], y=features.iloc[:, 3], 
                                      hue=df['Cluster'], palette="Set2", ax=ax5, s=50, alpha=0.7)
            ax5.set_title(f'Customer Clusters (K={n_clusters})')
            ax5.set_xlabel(cluster_features[0])
            ax5.set_ylabel(cluster_features[3] if len(cluster_features) > 3 else cluster_features[1])
            st.pyplot(fig5)
        
        with col2:
            # Pair plot for clusters
            fig_pair, ax_pair = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=features.iloc[:, 0], y=features.iloc[:, 1], 
                           hue=df['Cluster'], palette="Set2", ax=ax_pair, s=50, alpha=0.7)
            ax_pair.set_title(f'Customer Clusters - Alternative View')
            st.pyplot(fig_pair)
        
        # Cluster Summary
        st.subheader("Cluster Summary Statistics")
        cluster_summary = df.groupby('Cluster').agg({
            'Purchase': ['mean', 'median', 'std', 'count'],
            'Age_Encoded': 'mean',
            'Occupation': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A',
            'Marital_Status': 'mean'
        }).round(2)
        cluster_summary.columns = ['Avg Purchase', 'Median Purchase', 'Std Purchase', 
                                   'Count', 'Avg Age Group', 'Mode Occupation', 'Marital Rate']
        st.dataframe(cluster_summary)
        
        # Label clusters interactively
        st.subheader("Cluster Labels")
        st.write("Based on the cluster characteristics, you can assign meaningful labels:")
        cluster_labels = {}
        for i in range(n_clusters):
            label = st.text_input(f"Label for Cluster {i}:", 
                                  value=f"Cluster {i}", key=f"cluster_{i}")
            cluster_labels[i] = label
        
        # Display labeled clusters
        df['Cluster_Label'] = df['Cluster'].map(cluster_labels)
        st.write(df.groupby('Cluster_Label')['Purchase'].describe().round(2))
        
    else:
        st.warning("Please select at least 2 features for clustering.")

    # =============================================
    # STAGE 5: ASSOCIATION RULE MINING
    # =============================================
    st.header("Stage 5: Association Rule Mining")
    
    st.write("Discovering product combinations frequently bought together")
    
    # Prepare data for association rules
    basket = df[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']].copy()
    basket = basket.astype(str)
    
    # Create one-hot encoded basket
    basket_encoded = pd.get_dummies(basket)
    
    # Parameters
    min_support = st.slider("Minimum Support:", min_value=0.01, max_value=0.1, value=0.02, step=0.01)
    min_confidence = st.slider("Minimum Confidence:", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
    
    # Generate frequent itemsets
    frequent_items = apriori(basket_encoded, min_support=min_support, use_colnames=True)
    
    if len(frequent_items) > 0:
        st.subheader("Frequent Itemsets")
        st.write(f"Found {len(frequent_items)} frequent itemsets")
        st.dataframe(frequent_items.sort_values('support', ascending=False).head(20))
        
        # Generate association rules
        rules = association_rules(frequent_items, metric="confidence", 
                                  min_threshold=min_confidence, num_itemsets=len(frequent_items))
        
        if len(rules) > 0:
            st.subheader("Association Rules")
            
            # Filter by lift
            rules = rules[rules['lift'] >= 1.0]
            rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
            rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
            rules_display = rules_display.sort_values('lift', ascending=False)
            
            st.write(f"Found {len(rules_display)} association rules")
            st.dataframe(rules_display.head(20))
            
            # Visualize top rules
            st.subheader("Top 10 Rules by Lift")
            fig_rules, ax_rules = plt.subplots(figsize=(12, 6))
            top_rules = rules_display.head(10)
            bars = ax_rules.barh(range(len(top_rules)), top_rules['lift'].values, color='steelblue')
            ax_rules.set_yticks(range(len(top_rules)))
            ax_rules.set_yticklabels([f"{a} → {c}" for a, c in zip(top_rules['antecedents'], top_rules['consequents'])])
            ax_rules.set_xlabel('Lift')
            ax_rules.set_title('Top 10 Association Rules by Lift')
            plt.tight_layout()
            st.pyplot(fig_rules)
        else:
            st.warning("No association rules found with the current parameters. Try lowering the minimum confidence.")
    else:
        st.warning("No frequent itemsets found. Try lowering the minimum support.")

    # =============================================
    # STAGE 6: ANOMALY DETECTION
    # =============================================
    st.header("Stage 6: Anomaly Detection (Unusual Spenders)")
    
    # Method selection
    anomaly_method = st.radio("Select Anomaly Detection Method:", 
                              ["Isolation Forest", "Z-Score", "IQR Method"])
    
    if anomaly_method == "Isolation Forest":
        contamination = st.slider("Contamination Rate:", min_value=0.01, max_value=0.1, value=0.02, step=0.01)
        iso = IsolationForest(contamination=contamination, random_state=42)
        df['Anomaly'] = iso.fit_predict(df[['Purchase']])
        df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})  # 1 = anomaly, 0 = normal
        
    elif anomaly_method == "Z-Score":
        z_threshold = st.slider("Z-Score Threshold:", min_value=2.0, max_value=4.0, value=3.0, step=0.1)
        df['Z_Score'] = np.abs(stats.zscore(df['Purchase']))
        df['Anomaly'] = (df['Z_Score'] > z_threshold).astype(int)
        
    else:  # IQR Method
        Q1 = df['Purchase'].quantile(0.25)
        Q3 = df['Purchase'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['Anomaly'] = ((df['Purchase'] < lower_bound) | (df['Purchase'] > upper_bound)).astype(int)
        st.write(f"IQR Bounds: Lower = {lower_bound:.2f}, Upper = {upper_bound:.2f}")
    
    # Get anomalies
    anomalies = df[df['Anomaly'] == 1]
    normal = df[df['Anomaly'] == 0]
    
    st.subheader("Anomaly Detection Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Anomalies Detected", f"{len(anomalies):,}")
    with col3:
        st.metric("Anomaly Rate", f"{len(anomalies)/len(df)*100:.2f}%")
    
    # Visualize anomalies
    st.subheader("Anomaly Visualization")
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    ax6.scatter(normal.index, normal['Purchase'], c='blue', alpha=0.3, label='Normal', s=10)
    ax6.scatter(anomalies.index, anomalies['Purchase'], c='red', alpha=0.7, label='Anomaly', s=30)
    ax6.set_xlabel('Index')
    ax6.set_ylabel('Purchase Amount')
    ax6.set_title('Anomaly Detection Results')
    ax6.legend()
    st.pyplot(fig6)
    
    # Anomaly details
    st.subheader("Unusual High Spenders Details")
    st.dataframe(anomalies[['User_ID', 'Product_ID', 'Purchase', 'Age', 'Occupation', 'Gender']].head(20))
    
    # Anomaly demographics
    st.subheader("Anomaly Demographics Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig_anom_age, ax_anom_age = plt.subplots()
        sns.countplot(x='Age', data=anomalies, ax=ax_anom_age, palette='Reds')
        ax_anom_age.set_title('Anomalies by Age Group')
        ax_anom_age.tick_params(axis='x', rotation=45)
        st.pyplot(fig_anom_age)
    with col2:
        fig_anom_gender, ax_anom_gender = plt.subplots()
        gender_anom = anomalies['Gender'].value_counts()
        ax_anom_gender.pie(gender_anom.values, labels=gender_anom.index, autopct='%1.1f%%', colors=['lightblue', 'pink'])
        ax_anom_gender.set_title('Anomalies by Gender')
        st.pyplot(fig_anom_gender)

    # =============================================
    # STAGE 7: INSIGHTS & REPORTING
    # =============================================
    st.header("Stage 7: Key Insights & Recommendations")
    
    st.subheader("📊 Data-Driven Insights")
    
    # Calculate insights
    top_age_group = df.groupby('Age')['Purchase'].mean().idxmax()
    top_age_value = df.groupby('Age')['Purchase'].mean().max()
    
    gender_spending = df.groupby('Gender')['Purchase'].mean()
    higher_spender_gender = gender_spending.idxmax()
    
    top_product_cat = df['Product_Category_1'].value_counts().index[0]
    
    insight_list = [
        f"**Age Group Spending:** The **{top_age_group}** age group spends the most on average (${top_age_value:.2f}).",
        f"**Gender Analysis:** **{higher_spender_gender}** customers tend to spend more on average.",
        f"**Most Popular Category:** Product Category **{top_product_cat}** has the highest number of purchases.",
        f"**Anomaly Detection:** {len(anomalies)} unusual high spenders were identified ({len(anomalies)/len(df)*100:.2f}% of transactions).",
        f"**Customer Segments:** {n_clusters} distinct customer clusters were identified through K-Means clustering.",
        f"**Association Rules:** Product combinations were discovered that can inform cross-selling strategies."
    ]
    
    for insight in insight_list:
        st.markdown(f"• {insight}")
    
    st.subheader("💡 Business Recommendations")
    recommendations = [
        "**Target Marketing:** Focus marketing efforts on the highest-spending age groups with personalized offers.",
        "**Cross-Selling:** Use association rules to create bundle deals for frequently co-purchased products.",
        "**Premium Customer Focus:** Develop loyalty programs for the anomaly-detected high-value customers.",
        "**Category Optimization:** Stock more products from popular categories and consider expanding in those areas.",
        "**Personalized Promotions:** Use cluster labels to tailor promotions for different customer segments."
    ]
    
    for rec in recommendations:
        st.markdown(f"• {rec}")

else:
    st.info("Please upload a CSV file to begin the analysis.")
    
    # Show expected format
    st.subheader("Expected Dataset Format")
    st.write("""
    The dataset should contain the following columns:
    - **User_ID**: Unique identifier for each user
    - **Product_ID**: Unique identifier for each product
    - **Gender**: Gender of the user (M/F)
    - **Age**: Age group of the user (0-17, 18-25, 26-35, 36-45, 46-50, 51-55, 55+)
    - **Occupation**: Occupation code (masked)
    - **City_Category**: Category of the city (A, B, C)
    - **Stay_In_Current_City_Years**: Number of years stayed in current city
    - **Marital_Status**: Marital status (0/1)
    - **Product_Category_1, 2, 3**: Product categories
    - **Purchase**: Purchase amount
    """)
