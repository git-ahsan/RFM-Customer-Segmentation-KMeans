# Retail Customer Segmentation Using RFM Analysis and KMeans Clustering (Ongoing Project) 

## 📄 1. Project Summary

This project focuses on segmenting retail customers using **RFM analysis** (Recency, Frequency, Monetary) combined with **KMeans clustering**. By analyzing customer purchase behavior from a transactional dataset, distinct customer segments were identified and labeled (e.g., Diamond, Gold, Silver, Bronze). The goal is to help businesses implement targeted marketing strategies, improve customer retention, and maximize profitability using data-driven insights.

---

## 🎯 2. Project Objective

The primary objective of this project is to:

📊 Understand customer behavior through **RFM metrics**

✂️ Segment customers based on their transactional patterns

🔍 Apply **KMeans clustering** to identify hidden groups

📈 Visualize and interpret each segment

🚀 Deliver actionable recommendations for **personalized marketing and customer relationship management**

---

## 🛠️ 3. Tools & Technologies Used
The project was developed using the following tools and libraries:
| Category             | Tools/Technologies                 |
| :-----------------   | :--------------------------------- |
| Programming          | Python                             |
| Data Manipulation    | Pandas                             |
| Numerical operations | NumPy                              |
| Data Visualization   | Matplotlib, Seaborn                |
| Machine Learning     | Scikit-learn (KMeans), Yellowbrick |
| IDE                  | Jupyter Notebook                   |

---
## 📊 4. Dataset Description

The dataset used in this project is transactional retail data collected between **2009 and 2010**. It contains information about customer purchases, including invoice details, product quantities, unit prices, and customer IDs.
Key characteristics:
- **Total Records:** ~500,000 transactions
- **File Name:** [Retail Customer Data 2009-10.csv](https://github.com/git-ahsan/RFM-Customer-Segmentation-KMeans/blob/main/Retail%20Customer%20Data%202009-10.zip)
- **Format:** CSV

### 📁 Dataset Features:
| Column Name     | Description                                         |
|------------------|-----------------------------------------------------|
| Invoice No       | Unique identifier for each transaction             |
| Invoice Date     | Date of the transaction                            |
| Quantity         | Number of items purchased                          |
| Unit Price       | Price per item                                     |
| Customer ID      | Unique ID assigned to each customer                |
| Country          | Country where the transaction occurred             |

- ✅ The dataset contains both numeric and categorical features.
- ⚠️ Missing values were handled by dropping rows where `Customer ID` was null.
- 📌 This dataset is well-suited for RFM analysis because it includes **purchase dates**, **quantities**, and **prices**, which are essential for calculating Recency, Frequency, and Monetary values.

---

## 📈 5. Methodology / Workflow

This project follows a structured pipeline to perform customer segmentation using RFM analysis and KMeans clustering:

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

### 🔹 Step 4: Convert Invoice Date and Set Reference Date

To prepare for RFM (Recency, Frequency, Monetary) analysis, we must ensure the `Invoice Date` column is in datetime format. We also set a fixed reference date (`CurrentDate`) to calculate **Recency** later on.

###### Used .tail() to inspect the last few transactions
```python
df.tail(5)
```

###### Set a custom Current Date to calculate how recently each customer made a purchase
```python
CurrentDate = pd.to_datetime("2011-01-01")
CurrentDate
```

###### Converted Invoice Date from object to datetime64[ns] using pd.to_datetime(). Then Verified the conversion using .dtypes.
```python
df["Invoice Date"] = pd.to_datetime(df["Invoice Date"])
df.dtypes
```

### 🔹 Step 5: Identify and Remove Missing Customer IDs

Before performing RFM analysis, we must remove incomplete records. Customer ID is critical for segmentation, so entries without it are excluded.

###### At first checked initial dataset shape using .shape
```python
df.shape
```

###### Identified missing values in all columns using .isnull().sum(), where Found that some records had missing Customer ID and Product Description.
```python
df.isnull().sum()
```

###### Removed only those Customer ID records using dropna() and Re-checked dataset shape to confirm the number of valid entries
```python
df = df.dropna(subset=["Customer ID"])
df.shape
```

### 🔹 Step 6: Calculate `Recency` – Days Since Last Purchase

Recency refers to how recently a customer made a purchase. In this step, we calculate the number of days since each customer's last transaction based on a fixed reference date.

###### Grouped data by `Customer ID` and extracted the most recent `Invoice Date`
```python
Lastrnxdate = df.groupby(["Customer ID"]).max()[["Invoice Date"]]
Lastrnxdate
```

###### Calculated `Invoice Age` by subtracting the last purchase date from the `CurrentDate`
```python
Lastrnxdate["Invoice Age"] = (CurrentDate - Lastrnxdate["Invoice Date"]).dt.days
Lastrnxdate
```

###### Drop `Invoice Date` and Renamed the result as `Recency`, which represents the number of days since a customer's last transaction
```python
Recency = Lastrnxdate.drop("Invoice Date", axis=1)
```

### 🔹 Step 7: Calculate Frequency – Number of Purchases

Frequency indicates how many times a customer made a purchase. To avoid counting the same invoice multiple times, we use unique invoice numbers for each customer.

###### At First Checked the original dataset
```python
df
```

###### Grouped by Customer ID and counted the number of unique invoices per customer
```python
UniqueInvoice = df.drop_duplicates(subset="Invoice No")
UniqueInvoice
```

###### Grouped by `Customer ID` to count the number of unique invoices and create the `Frequency` DataFrame representing each customer's total purchases.
```python
Frequency = UniqueInvoice.groupby(["Customer ID"]).count()[["Invoice No"]]
Frequency
```

### 🔹 Step 8: Calculate Monetary – Total Amount Spent

Monetary refers to the total amount a customer has spent. This step involves calculating the transaction value per row and summing it by customer.

###### Ensures we're working with a safe, independent copy of the DataFrame. This avoids the `SettingWithCopyWarning` from pandas when modifying a sliced DataFrame.
```python
df = df.copy()
```

###### Calculates the `Monetary` value of each transaction and stores it in a new column called "Total".
```python
df.loc[:, "Total"] = df["Quantity"] * df["Unit Price"]
```

###### Grouped by `Customer ID` and summed up the "Total" column to create the Monetary DataFrame. This gives the total spending per customer across all their purchases.
```python
Monetary = df.groupby(["Customer ID"])[["Total"]].sum()
Monetary
```

### 🔹 Step 9: Combine Recency, Frequency, and Monetary into RFM Table

After calculating the three individual metrics, we combine them into a single DataFrame for customer segmentation analysis.

###### Used pd.concat() to join the Recency, Frequency, and Monetary DataFrames side-by-side
```python
RFM = pd.concat([Recency, Frequency, Monetary], axis=1)
```

###### Renamed columns for clarity and standard naming convention
```python
RFM.columns = ['Recency', 'Frequency', 'Monetary']
RFM
```

### 🔹 Step 10: Scale the RFM Features for Clustering

Before applying KMeans clustering, it's essential to standardize the RFM values to ensure that each feature contributes equally to the distance calculations.
This step utilizes `StandardScaler` from `sklearn.preprocessing` to perform the scaling.

RFM features have different units and value ranges (e.g., Recency in days, Monetary in currency). If left unscaled, features with larger numerical ranges would dominate the clustering process. Since KMeans relies on Euclidean distance to group data points, this could bias the results.

Standardization solves this issue by transforming all features to have a mean of 0 and a standard deviation of 1, allowing for fair and balanced clustering.

```python
scaler = StandardScaler()
scaled = scaler.fit_transform(RFM)
scaled
scaled.shape
```

### 🔹 Step 11: Determine Optimal k Using the Elbow Method

To find the ideal number of customer segments (clusters), I use the **Elbow Method**, which plots the **Sum of Squared Errors (SSE)** against a range of possible `k` values. The point where the SSE curve starts to flatten (like an elbow) is considered the optimal value of `k`.
This step utilizes `KMeans` from `sklearn.cluster`.

```python
k_range = range(2, 11)
sse = []  # Sum of Squared Errors

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled)
    sse.append(km.inertia_)
sse
```

### 🔹 Step 12: Visualize Elbow Curve Using Matplotlib

Before applying `KElbowVisualizer`, I plotted the Elbow Curve manually using `matplotlib.pyplot` to get a clear visual of how the **Sum of Squared Errors (SSE)** changes with different `k` values.

```python
fig, axes = plt.subplots(figsize=(8,6), facecolor="#E6F9FA")

axes.plot(
    k_range, sse,
    color="purple",
    lw=3,
    ls='--',
    marker='s',
    markersize=8,
    markerfacecolor="yellow",
    markeredgewidth=3,
    markeredgecolor="red"
)

plt.grid(True, linestyle='--', alpha=0.8)
plt.tight_layout()
plt.show()
```
#### 📷 Preview

![Preview](https://github.com/git-ahsan/RFM-Customer-Segmentation-KMeans/blob/main/Visualization%20Charts/Elbow%20Curve.jpg)

### 🔹 Step 13: Visualize Optimal k Using KElbowVisualizer (Yellowbrick)

To complement the manual elbow plot, I used **Yellowbrick's KElbowVisualizer** to automatically identify the optimal number of clusters (`k`) based on **distortion (SSE)**. 
This step utilizes both `KElbowVisualizer` from `yellowbrick.cluster` and `KMeans` from `sklearn.cluster` to perform the visualizer.

```python
model = KMeans(random_state=42)
elbow = KElbowVisualizer(
    model,
    k=(2, 10),
    metric='distortion',
    timings=True,           # Shows training time per k
    locate_elbow=True,      # Automatically marks the elbow point
    size=(700, 500)         # Resize the plot
)

elbow.fit(scaled)
elbow.show()
```
#### 📷 Preview

![Preview](https://github.com/git-ahsan/RFM-Customer-Segmentation-KMeans/blob/main/Visualization%20Charts/KElbowVisualizer.jpg)

### 🔹 Step 14: Fit KMeans Model with Optimal Number of Clusters-k

The KMeans model was instantiated with 4 clusters, reflecting the elbow analysis. A `random_state of 42` was set to ensure consistent results across all executions. The model was then trained on the scaled RFM data to delineate distinct customer segments.

```python
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled)
```

### 🔹 Step 15: Assign Cluster Labels to Customers

After training the final KMeans model, I assigned each customer to a cluster by attaching the predicted labels to the RFM table.

###### Used kmeans.labels_ to extract cluster predictions for each customer.
```python
kmeans.labels_
```

###### Verified the number of labels matched the number of customers.
```python
kmeans.labels_.shape
```

###### Added a new column `Clusters` to the RFM DataFrame to store the segment each customer belongs to.
```python
RFM["Clusters"] = kmeans.labels_  # Assign labels to RFM table
RFM
```

###### Checked the distinct cluster labels (e.g., 0, 1, 2, 3)
```python
RFM.Clusters.unique()
```

### 🔹 Step 16: Visualize Customer Segments in 3D (RFM Clusters)

To better understand the distribution of customer clusters, a 3D scatter plot was generated to visualize customer clusters based on their **Recency**, **Frequency**, and **Monetary** (RFM) values. This plot, created using matplotlib's 3D projection, maps Recency to the X-axis, Frequency to the Y-axis, and Monetary to the Z-axis. Each customer cluster is differentiated by color using the "Accent" colormap, allowing for a clear visual separation of segments across all three RFM dimensions. The axes are labeled and a grid is enabled to enhance readability, ultimately helping to validate the quality of the customer clustering.

```python
fig = plt.figure(figsize=(12, 9), facecolor="#E6F9FA")   ###Creates Matplotlib figure with a light background.
ax = fig.add_subplot(111, projection='3d', label="bla")  ###adds a 3D subplot to the figure.

###creates a 3D scatter plot of RFM data.
scatter = ax.scatter(
    RFM["Recency"], 
    RFM["Frequency"], 
    RFM["Monetary"], 
    c=RFM["Clusters"], 
    cmap="Accent", 
    s=60, 
    edgecolor='k', 
    alpha=0.8
)

###Titles and labels the 3D plot axes.
ax.set_title("3D Clustering of Customers (RFM)", fontsize=16, fontweight='bold')
ax.set_xlabel("Recency", fontsize=12, fontweight='bold')
ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
ax.set_zlabel("Monetary", fontsize=12, fontweight='bold')
ax.grid(True)            #Titles and labels the 3D plot axes.

plt.tight_layout()
plt.show()
```
#### 📷 Preview

![Preview](https://github.com/git-ahsan/RFM-Customer-Segmentation-KMeans/blob/main/Visualization%20Charts/3D.jpg)

### 🔹 Step 17: Analyze and Profile Each Customer Cluster

After clustering, each segment is analyzed to uncover customer behavior patterns based on Recency, Frequency, and Monetary values. By filtering the RFM dataset for each cluster (0, 1, 2, 3), distinct profiles can be identified — such as customers with low recency and high monetary value (likely loyal or VIPs), those with high recency and low frequency (possibly inactive or at-risk), and others with moderate values (average or developing customers). This interpretation helps transform raw clusters into meaningful business personas for data-driven marketing strategies.

```python
RFM[RFM.Clusters == 0]
RFM[RFM.Clusters == 1]
RFM[RFM.Clusters == 2]
RFM[RFM.Clusters == 3]
```

### 🔹 Step 18: Summarize RFM Metrics by Cluster

To better understand each customer segment, the average Recency, Frequency, and Monetary values were calculated for every cluster. This aggregated view reveals the overall behavior of each group — such as how recently customers purchased, how often they buy, and how much they typically spend. These cluster-wise summaries serve as the foundation for assigning intuitive labels like “VIP,” “At-Risk,” or “Potential Loyalists,” making the segmentation actionable for strategic business decisions.

```python
final = RFM.groupby("Clusters").mean()[["Recency", "Frequency", "Monetary"]]
final
```

### 🔹 Step 19: Assign Descriptive Labels to Customer Segments

To make the cluster segments more interpretable for business users, numeric cluster labels were mapped to descriptive names — such as “Diamond,” “Gold,” “Silver,” and “Bronze” — based on their average RFM scores. A custom function was applied row-wise to assign each customer to a group, enabling clearer communication of customer value and facilitating targeted marketing strategies aligned with each segment’s behavior.

```python
def func(row):
    if row["Clusters"] == 2:
        return 'Diamond'
    elif row["Clusters"] == 3:
        return 'Gold'
    elif row["Clusters"] == 0:
        return 'Silver'
    else:
        return 'Bronze'

RFM['Group'] = RFM.apply(func, axis=1)
RFM
```

### 🔹 Step 20: Count Customers in Each Segment

To understand how customers are distributed across the defined segments, the number of customers in each group (Diamond, Gold, Silver, Bronze) was calculated using value_counts(). The result was converted into a DataFrame and reset with a new index, making it suitable for reporting, visualization, and further analysis. This summary helps evaluate which segments are dominant and where targeted marketing or improvement strategies should focus.

```python
result = pd.DataFrame(RFM.Group.value_counts())
result = result.reset_index()
result
```

### 🔹 Step 21: Visualize Customer Segmentation Results

To visually represent the results of customer segmentation, a combined chart was created using Matplotlib and Seaborn. The bar chart (on a log scale) shows the absolute count of customers per group, while the donut chart displays the percentage share of each segment. Custom colors and annotations were applied for clarity and visual appeal. Together, these charts provide a clear, interpretable overview of how customers are distributed across the Diamond, Gold, Silver, and Bronze segments.

```python
Group = result["Group"]
Count = result["count"]
Customcolors = ["#81DED0", "#FA667E", "#FED315", "#461856"]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar Chart
sns.barplot(x=Group, y=Count, hue=Group, palette=Customcolors, ax=ax1, legend=False)
ax1.set_yscale('log')
ax1.set_title("Bar Chart (Log Scale)", fontsize=16, fontweight='bold')
ax1.set_xlabel("Customer Group", fontsize=12, fontweight='bold')
ax1.set_ylabel("Log Count", fontsize=12, fontweight='bold')

for container in ax1.containers:
    ax1.bar_label(container, fontsize=10, fontweight='bold')

# Donut (Pie) Chart
wedges, texts, autotexts = ax2.pie(
    Count,
    labels=Group,
    colors=Customcolors,
    autopct='%1.1f%%',
    startangle=0,
    wedgeprops=dict(width=0.6)
)
ax2.set_title("Donut Chart (Percentage)", fontsize=16, fontweight='bold')
plt.setp(autotexts, size=10, weight="bold", color="Black")

plt.suptitle("Customer Group Distribution Analysis", fontsize=20, fontweight='bold', color='navy')
plt.tight_layout()
plt.show()
```
#### 📷 Preview

![Preview](https://github.com/git-ahsan/RFM-Customer-Segmentation-KMeans/blob/main/Visualization%20Charts/Customer%20Group%20Distribution.png)

### 🔹 Step 22: Identify Highest and Lowest Monetary Value Customers by Segment

To explore spending extremes within customer segments, the dataset was filtered by cluster and sorted by the Monetary value. This helped identify both the highest-value and lowest-value customers within specific groups, such as Cluster 2 (Diamond) and Cluster 1 (Bronze). These insights are useful for designing personalized loyalty programs, upselling strategies, or retention efforts targeting high- and low-value customers individually.

```python
### Analyze Cluster 2 (Diamond)
ABC = RFM[RFM.Clusters == 2]
ABC[ABC['Monetary'] == ABC['Monetary'].min()]  # Lowest spender in Cluster 2 (Diamond)
ABC[ABC['Monetary'] == ABC['Monetary'].max()]  # Highest spender in Cluster 2 (Diamond)

# Analyze Cluster 1 (Bronze)
ABC = RFM[RFM.Clusters == 1]
ABC[ABC['Monetary'] == ABC['Monetary'].min()]  # Lowest spender in Cluster 1 (Bronze)
ABC[ABC['Monetary'] == ABC['Monetary'].max()]  # Highest spender in Cluster 1 (Bronze)
```
---
## 📌 6. Results & Visualizations

The RFM analysis and KMeans clustering algorithm successfully segmented the retail customers into four distinct groups: **Diamond, Gold, Silver, and Bronze**. These segments were based on their Recency (how recently they purchased), Frequency (how often they purchased), and Monetary (how much they spent) behavior.

Key visualizations were created to illustrate and support the results:

- 🎯 **3D Cluster Plot**: Showcased how customers are grouped in the RFM space, validating that the clusters are well-separated.
- 📊 **Bar Chart (Log Scale)**: Displayed the customer count in each group, scaled logarithmically to better show group differences.
- 🍩 **Donut Chart (Percentage)**: Visualized the proportional distribution of customers across segments.
- 📋 **Cluster Profile Table**: Showed average RFM values for each cluster, helping identify high-value vs. low-value segments.

These visual insights confirmed that:
- **Diamond customers** (Cluster 2) are highly valuable: they purchase frequently, spend more, and return often.
- **Gold and Silver groups** reflect moderate customer behavior and represent potential upsell targets.
- **Bronze customers** (Cluster 1) are least engaged — strategies like reactivation campaigns or offers may improve retention.

The visual tools helped interpret and communicate the clustering results clearly for stakeholders or business use cases.

---
## 💡 7. Insights & Business Recommendations

Based on the RFM-KMeans clustering results, the customer base can be divided into four distinct groups with clear behavioral traits. These insights provide valuable guidance for crafting targeted business strategies:

### 🔷 Diamond Segment
- **Profile**: Recent, frequent, and high spenders — the most loyal and profitable customers.
- **Strategy**: Focus on **retention** and **rewards**. Offer exclusive perks, early access to products, or loyalty programs to reinforce brand loyalty.

### 🟡 Gold Segment
- **Profile**: Moderate in recency and spending, but still active.
- **Strategy**: Target with **upselling** and **personalized offers**. Encourage higher spending with bundles, cross-selling, or personalized recommendations.

### ⚪ Silver Segment
- **Profile**: Regular buyers with lower spending and moderate recency.
- **Strategy**: Use **re-engagement** tactics. Provide discounts or limited-time offers to increase frequency and monetary value.

### 🟤 Bronze Segment
- **Profile**: Infrequent buyers with older purchases and lower overall spending.
- **Strategy**: Identify possible causes (e.g., churn or disinterest). Use **email campaigns**, **surveys**, or **reactivation incentives** to win them back.

### 📈 Overall Business Actions
- Allocate marketing budget based on segment value.
- Design tiered loyalty programs using the RFM tiers.
- Track segment shifts over time to monitor retention and churn.

> 📌 These actionable insights enable data-driven decision-making to improve customer satisfaction, lifetime value, and overall profitability.

---
## 📁 8. Repository Structure

The repository contains all relevant files and notebooks used to perform Retail Customer Segmentation using RFM analysis and KMeans clustering.

├── [RFM.ipynb](https://github.com/git-ahsan/RFM-Customer-Segmentation-KMeans/blob/main/RFM.ipynb) # Jupyter Notebook with full RFM & KMeans workflow

├── Retail Customer Data 2009-10.csv # Original dataset used for analysis

├── images/ # Folder for visualizations (optional: for use in README or reports)

├── README.md # Project documentation


### 📌 Notes:
- The main analysis is performed in `RFM.ipynb`, following a step-by-step methodology from data preprocessing to model training and visualization.
- Ensure required libraries (e.g., pandas, matplotlib, seaborn, scikit-learn, yellowbrick) are installed before running the notebook.
---
## 🧠 9. Key Learnings / What I Learned

This project provided hands-on experience in applying data science techniques to solve a real-world business problem — customer segmentation. Key takeaways include:

- 📊 Gained practical understanding of **RFM (Recency, Frequency, Monetary) Analysis** and its importance in identifying customer behavior patterns.
- 📦 Learned how to **preprocess and clean transactional data**, including handling null values, date conversion, and feature engineering.
- 🤖 Applied **KMeans Clustering** effectively after standardizing features, and understood the significance of choosing the right number of clusters using the **Elbow Method** and **KElbowVisualizer**.
- 🎯 Practiced converting raw cluster outputs into **actionable business segments** and interpreting them for decision-making.
- 📈 Improved my skills in **visual storytelling**, using 3D plots, bar charts, and donut charts to present insights clearly.
- 💡 Developed confidence in **end-to-end project handling** — from data loading and exploration to modeling, visualization, and documentation.

> This project deepened my confidence in data science and strengthened both my technical and business analytical thinking.
---
## 📬 Contact

**Md. Ahsan Ul Islam**  
🎓 Junior Data Analyst  
🔗 [LinkedIn](https://www.linkedin.com/in/md-ahsan-ul-islam)  
🔗 [GitHub](https://github.com/git-ahsan)

---

## 🏷️ Tags

`#Python` `#DataScience` `#BusinessIntelligence` `#MachineLearning` `#RFM` `#CustomerSegmentation` `#ScikitLearn` `#GitHubPortfolio` `#DataManipulation` `#Pandas` `#Numericaloperations` `#NumPy` `#DataVisualization` `#Matplotlib` `#Seaborn` `#KMeans` `#Yellowbrick` `#JupyterNotebook`
