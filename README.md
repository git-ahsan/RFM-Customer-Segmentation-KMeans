# Retail Customer Segmentation Using RFM Analysis and KMeans Clustering (Ongoing Project) 

This project implements **RFM (Recency, Frequency, Monetary) Analysis** along with **KMeans Clustering** to segment customers based on their purchasing behavior. The analysis helps businesses identify valuable customer groups, improve marketing strategies, and enhance customer retention.

---

## 🎯 Project Objective
📊 To understand customer purchase behavior using historical transaction data.

👥 To calculate RFM metrics and use them to cluster customers into distinct segments.

💎 To identify high-value customers and offer actionable insights for business decision-making.

📈 To visualize the clustering results and support data-driven marketing strategies.

---

## 🛠️ Tools & Technologies Used

| Category             | Tools/Technologies                 |
| :-----------------   | :--------------------------------- |
| Programming          | Python                             |
| Data Manipulation    | Pandas, NumPy                      |
| Visualization        | Matplotlib, Seaborn                |
| Machine Learning     | Scikit-learn (KMeans), Yellowbrick |
| IDE                  | Jupyter Notebook / Google Colab    |

---

## 📊 Dataset Description

📁 CSV File: [Retail Customer Data]

The dataset used in this project is a **retail transaction dataset** containing historical sales data from an online retailer between **December 1, 2009** and **December 9, 2010**.

Key characteristics:

- **Total Records:** ~500,000 transactions
- **File Name:** [Retail Customer Data 2009-10.csv](https://github.com/git-ahsan/RFM-Customer-Segmentation-KMeans/blob/main/Retail%20Customer%20Data%202009-10.zip)
- **Format:** CSV

### Features:

| Column Name     | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `Invoice No`     | Unique identifier for each transaction                                      |
| `Stock Code`     | Product (item) code                                                         |
| `Product Description`   | Product name                                                                |
| `Quantity`      | The number of items purchased per transaction                               |
| `Invoice Date`   | Date and time when the invoice was generated                                |
| `Unit Price`     | Price per unit of the product                                               |
| `Customer ID`    | Unique customer identifier                                                  |
| `Country`       | Country name where the customer resides                                     |

> 🛠️ Note: Rows with missing `CustomerID` were dropped to ensure clean customer segmentation.

From this dataset, **RFM (Recency, Frequency, Monetary)** features were derived to perform customer segmentation analysis using clustering techniques.

---

## 📈 Methodology / Workflow

This project follows a structured pipeline to perform customer segmentation using RFM analysis and KMeans clustering:

---

### 🔹 Step 1: Importing Required Libraries

In this step, we import all the essential Python libraries and modules required for data loading, analysis, visualization, and clustering.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
```

### 🔹 Step 2: Load and Inspect the Dataset

The dataset is loaded using `pandas.read_csv()` with appropriate encoding to handle special characters. Initial inspection is done to understand the structure and shape of the data.

```python
df = pd.read_csv("Retail Customer Data 2009-10.csv", encoding='unicode_escape')
df
```

### 🔹 Step 3: Check Data Types of Each Column

Before performing any transformation or analysis, it's important to understand the data types of each column in the dataset. This ensures the right operations are applied to each feature.

```python
df.dtypes
```

