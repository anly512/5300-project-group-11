# UK Online Retail Data Analysis

## Project Overview
This project involves a statistical analysis of transactional data from a UK-based online retail store aimed at enhancing customer-centric business intelligence. The primary focus is on customer segmentation using various data mining techniques to identify key customer groups and their characteristics, which can be leveraged to optimize marketing strategies.

## Data Source
The analysis is conducted on [https://archive.ics.uci.edu/dataset/352/online+retail](https://archive.ics.uci.edu/dataset/352/online+retail). The dataset includes transactional data with attributes such as recency, frequency, and monetary value of purchases.

## Methodology

### Data Preprocessing
- **RFM Analysis**: A marketing analysis tool used to identify customer value is implemented. It segments the data based on Recency, Frequency, and Monetary value, commonly referred to as RFM.

### Exploratory Data Analysis
- **Feature Engineering**: Techniques like PCA (Principal Component Analysis) are used to reduce dimensions and extract relevant features.

### Model Building
- **Binary Classification**: To identify 'High Spenders', four predictive models are employed: Logistic Regression, LDA, QDA, and Ridge Classification.
- **K-means Clustering**: Used for customer segmentation into several groups based on transactional behavior.

### Model Evaluation
- Models are evaluated based on accuracy, precision, sensitivity, and specificity, with detailed ROC curves provided to compare model performance.

## Results

### Key Findings
- Effective segmentation of customers into four distinct groups:
- **Type 1**: Frequent and high spending customers with recent transactions.
- **Type 2**: Customers with infrequent and low-value purchases.
- **Type 3**: Customers who shop frequently and spend generously, but with less recent purchases.
- **Type 4**: Customers preferring premium products and showing frequent recent activity.
- Ensemble Machine Learning models provide robust predictions with high accuracy, especially the Random Forest model which achieved a 99.6% accuracy rate.

## Conclusion
The project confirms the effectiveness of machine learning models in customer segmentation and classification, providing actionable insights for tailored marketing strategies.

## Contributors
- Wenbin Fang
- Zhaoqian Xue
- Michael Xie

## Acknowledgements
Thanks to Georgetown University DSAN for guidance and resources throughout this project.