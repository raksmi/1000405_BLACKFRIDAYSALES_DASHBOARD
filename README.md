🛍️ Black Friday Sales Analytics Dashboard

An AI-powered data mining project that analyzes Black Friday retail data to uncover customer purchasing patterns, identify shopper segments, discover product associations, and detect unusual spending behavior.
The project uses data mining techniques such as clustering, association rule mining, anomaly detection, and interactive visualization to generate actionable business insights.

📌 Project Objective

Retailers experience massive traffic and sales during Black Friday events. Understanding customer purchasing patterns is essential for improving marketing strategies and inventory planning.

This project aims to:

Analyze customer shopping behavior during Black Friday sales

Segment customers into groups based on purchasing habits

Discover relationships between products frequently purchased together

Detect unusual purchasing behavior (high spenders)

Present insights using an interactive Streamlit dashboard

📊 Dataset Description

The dataset used is the Black Friday Sales Dataset, which contains information about customer demographics, product categories, and purchase amounts.

Key Columns
Column	Description
User_ID	Unique identifier for each customer
Product_ID	Unique identifier for each product
Gender	Customer gender (M/F)
Age	Customer age group
Occupation	Occupation code
City_Category	City type (A, B, C)
Stay_In_Current_City_Years	Years spent in current city
Marital_Status	Marital status (0 = unmarried, 1 = married)
Product_Category_1	Main product category
Product_Category_2	Secondary product category
Product_Category_3	Third product category
Purchase	Amount spent by customer
🧹 Data Cleaning & Preprocessing

Before performing analysis, the dataset was cleaned and prepared:

Missing Values

Missing values in Product_Category_2 and Product_Category_3 were replaced with 0.

Encoding

Categorical variables were converted into numerical values:

Gender → M = 0, F = 1

Age groups mapped into ordered numbers

City category encoded using categorical codes

Stay years converted into numeric values

Feature Engineering

Additional encoded features created:

Gender_Encoded

Age_Encoded

City_Encoded

Stay_Encoded

These features were used for clustering and correlation analysis.

📈 Exploratory Data Analysis (EDA)

EDA helps visualize and understand the structure of the dataset.

Visualizations Created

Purchase distribution histogram

Box plots of purchase amount by age group

Gender spending comparison

City category spending analysis

Product category popularity charts

Feature correlation heatmap

These visualizations reveal spending trends, demographic behavior, and product demand patterns.

🎯 Customer Segmentation (Clustering)

Customer segmentation was performed using K-Means clustering.

Features Used

Age (encoded)

Occupation

Marital status

Purchase amount

Steps

Data standardization using StandardScaler

Optimal cluster selection using the Elbow Method

K-Means clustering applied

Customer segments visualized using scatter plots

Example Segments

🤑 Premium Spenders

🛍️ Regular Shoppers

💰 Budget Conscious Customers

This helps businesses target different customer groups with personalized marketing strategies.

🔗 Product Association Analysis

To understand which products are frequently bought together, Association Rule Mining was applied using the Apriori algorithm.

Process

Product categories converted into one-hot encoded basket format

Frequent itemsets discovered using Apriori

Association rules generated using metrics:

Support

Confidence

Lift

Example Insight

Customers buying products from Category A often purchase items from Category B as well.

This information can help retailers design bundle offers and cross-selling strategies.

🚨 Anomaly Detection

Anomaly detection was performed to identify unusual spending patterns.

Algorithm Used

Isolation Forest

Purpose

To detect customers whose purchases significantly differ from normal behavior.

Insights

Identification of high-value customers

Detection of abnormally large purchases

This information can be useful for:

VIP customer identification

Fraud detection

Targeted loyalty programs

💡 Key Insights

Some insights discovered from the analysis include:

Certain age groups spend more on average

Specific product categories dominate sales

Some customers show exceptionally high spending patterns

Customer clusters reveal different shopping behaviors

These insights can help businesses optimize marketing strategies and product placement.

🚀 Streamlit Dashboard

The project includes an interactive Streamlit web application that allows users to explore insights visually.

Dashboard Features

📊 Sales overview and purchase distribution

🎯 Customer segmentation visualization

🔗 Product association insights

🚨 Anomaly detection dashboard

💡 AI-generated business recommendations

Users can interact with the dashboard and adjust parameters such as:

Number of clusters

Anomaly detection sensitivity

🛠 Technologies Used

Python

Streamlit

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Mlxtend

📂 Project Structure
BlackFridaySales-Dashboard
│
├── app.py
├── BlackFriday.csv
├── requirements.txt
├── logo.png
└── README.md
