# Retail Customer Segmentation Using RFM Analysis and KMeans Clustering

## 📄 1. Project Summary

This project focuses on **Retail Customer Segmentation using RFM Analysis and KMeans Clustering** to identify valuable customer groups and enable targeted marketing strategies.

Using historical retail transaction data, customers were segmented based on their:

- **Recency** – How recently a customer made a purchase  
- **Frequency** – How often a customer makes purchases  
- **Monetary** – How much money a customer spends

By applying the **RFM model** and clustering techniques, we discovered hidden patterns in customer behavior and classified them into distinct segments like **Diamond, Gold, Silver, and Bronze**. These insights help businesses:

- Personalize customer experiences
- Retain high-value customers
- Reactivate inactive users
- Optimize marketing resources

📊 This end-to-end project demonstrates a practical application of data science to solve real-world business challenges using Python, pandas, scikit-learn, and visual storytelling with Seaborn and Matplotlib.

---

## 🎯 2. Project Objective

The primary objective of this project is to:

📊 Understand customer behavior through **RFM (Recency, Frequency, Monetary) metrics**  
✂️ Segment customers based on their transactional patterns  
🔍 Apply **KMeans clustering** to identify hidden customer groups  
📈 Visualize and interpret each segment for business decision-making  
📌 Label segments (e.g., Diamond, Gold) to guide **marketing and retention strategies**  
🧪 Score and rank high-value customers to enable **personalized campaigns**

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

###### Two records contained null values in the `Customer ID` and `Product Description` fields. Since `Customer ID` is essential for customer-level analysis, only those rows with missing `Customer ID` were removed using `dropna()`. The dataset shape was then re-checked to confirm the number of valid entries.
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

### 🔹 Step 22: Deep Dive into Diamond Segment (Cluster 2)

To gain more insight into the high-value Diamond segment (Cluster 2), the dataset was filtered to explore customers at the two extremes of spending. This helped identify the highest spender and the least active spender within the group. Such outlier analysis can be valuable for personalized engagement strategies, loyalty programs, or even fraud checks in real-world applications.

```python
ABC = RFM[RFM.Clusters == 2]

# Identify customer with the lowest spending in the Diamond group
ABC[ABC['Monetary'] == ABC['Monetary'].min()]

# Identify customer with the highest spending in the Diamond group
ABC[ABC['Monetary'] == ABC['Monetary'].max()]
```

### 🔹 Step 23: Extract Customers from Diamond Segment

To prepare for targeted marketing or personalized campaigns, the dataset was filtered to extract all customers assigned to the “Diamond” segment — those with the most frequent, recent, and high-spending behaviors. This subset represents the business’s top-tier customers and can be used for loyalty programs, VIP offers, or high-priority retention strategies.

```python
diamond_customers = RFM[RFM["Group"] == "Diamond"]
diamond_customers
```

### 🔹 Step 24: Normalize Diamond Segment RFM Values

To prepare the Diamond segment data for consistent comparison and visualization, the RFM values were scaled between 0 and 1 using **MinMaxScaler** from `sklearn.preprocessing`. This ensures that each feature (Recency, Frequency, and Monetary) contributes equally and is not skewed by differing ranges. Normalization is especially useful before plotting radar charts or when feeding the data into models or scoring systems that require uniform feature scales.

```python
scaler = MinMaxScaler()
normalized = scaler.fit_transform(diamond_customers[["Recency", "Frequency", "Monetary"]])
normalized
```

### 🔹 Step 25: Create DataFrame for Normalized RFM Values

The normalized RFM values for Diamond customers were converted into a new DataFrame named `diamond_normalized`, with appropriate column names and original customer indices preserved. This structure makes it easy to analyze, visualize, or merge with other data sources for deeper insights or targeted actions on high-value customers.

```python
diamond_normalized = pd.DataFrame(normalized, columns=["Recency_N", "Frequency_N", "Monetary_N"], index=diamond_customers.index)
diamond_normalized
```

### Step 26: Calculate Weighted Score for Diamond Customers  
To prioritize customers within the high-value Diamond segment, we computed a weighted score using normalized RFM values. Recency was inverted (lower values = better) and custom weights were applied:  
- **30% Recency** (prioritizing recent purchases),  
- **20% Frequency** (moderate emphasis on loyalty),  
- **50% Monetary** (maximum focus on spending behavior).  

This optimized weighting identifies top-tier Diamond customers for exclusive campaigns, balancing revenue potential with engagement.  

```python
diamond_customers = diamond_customers.copy()

diamond_customers["Score"] = (
    (1 - diamond_normalized["Recency_N"]) * 0.30 +  # Inverted for lower=better
    diamond_normalized["Frequency_N"] * 0.20 +
    diamond_normalized["Monetary_N"] * 0.50
)
```

### 🔹 Step 27: Rank Diamond Customers by Score

To finalize the prioritization of customers in the Diamond segment, a ranking column was added based on the calculated Score. Customers were ranked in descending order, with rank 1 representing the most valuable individual in the group. This ranking enables targeted engagement, top-tier reward programs, or focused retention efforts on the highest contributors.

```python
diamond_customers["Diamond_Rank"] = diamond_customers["Score"].rank(ascending=False).astype(int)
```

### 🔹 Step 28: Sort Diamond Customers by Rank

The `diamond_customers` DataFrame was sorted in ascending order by the `Diamond_Rank` column to place the **most valuable customers at the top**. This final view enables easy extraction of the top-tier customers for exclusive targeting, strategic engagement, or exporting to external tools like Power BI or CRM systems.

```python
diamond_customers_sorted = diamond_customers.sort_values("Diamond_Rank")
diamond_customers_sorted
```

### 🔹 Step 29: Final View of Ranked Diamond Customers

Displayed a clean, focused summary of the ranked Diamond customers, showing each customer’s priority rank, calculated score, and original RFM metrics. This final view enables stakeholders to quickly identify and understand the top-performing customers, making it ideal for reporting, executive dashboards, or direct action via marketing or loyalty initiatives.

```python
diamond_customers_sorted[["Diamond_Rank", "Score", "Recency", "Frequency", "Monetary"]]
```

### 🔹 Step 30: Visualize Ranked Diamond Customers with Score and ID

A bar chart was created to visually rank Diamond customers based on their composite RFM scores. Each bar represents a customer’s score, with the Customer ID displayed above the bar for easy identification. This plot offers a quick, intuitive overview of the top-tier customers and can be used in presentations, dashboards, or performance tracking reports to communicate which customers contribute the most value to the business.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Prepare the figure
plt.figure(figsize=(10, 6), facecolor="#F7FAFC")

Rank = diamond_customers_sorted["Diamond_Rank"]
Score = diamond_customers_sorted["Score"]

# Create barplot
barplot = sns.barplot(
    x=Rank,
    y=Score,
    hue=Rank,
    palette="viridis",
    legend=False
)

# Add Customer ID labels on top of bars
for index, row in diamond_customers_sorted.iterrows():
    barplot.text(
        x=row["Diamond_Rank"] - 1,
        y=row["Score"] + 0.01,
        s=str(index),
        ha='center',
        fontsize=10,
        fontweight='bold',
        color='black'
    )

# Styling
plt.title("Diamond Customers Ranked by Composite RFM Score", fontsize=16, fontweight='bold', color='navy')
plt.xlabel("Customer Rank", fontsize=12, fontweight='bold')
plt.ylabel("Score", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
```
![Preview](https://github.com/git-ahsan/RFM-Customer-Segmentation-KMeans/blob/main/Visualization%20Charts/Diamond.jpg)

---
## 6. 📊 Results & Visualizations

After performing RFM analysis and clustering using KMeans, we identified **4 distinct customer segments**: 🟣 Diamond, 🟡 Gold, 🔘 Silver, and 🔴 Bronze. Each segment represents different behavioral patterns based on Recency, Frequency, and Monetary value.

### 🧩 Segment Distribution:
- A **bar chart (log scale)** and a **donut chart** were used to illustrate the number and percentage of customers in each group.
- This helped identify which customer groups are the most dominant and which require attention.

### 💎 Diamond Segment (High Value):
- A deep dive was conducted into the **Diamond cluster**, including normalization and a custom **RFM-based scoring system**.
- Each Diamond customer was **ranked** based on their behavior using a weighted score (Recency: 25%, Frequency: 25%, Monetary: 50%).

### 📈 Visualizations:
- ✅ **Customer Segment Distribution** (Bar & Donut)
- ✅ **3D Cluster Plot** using Recency, Frequency, and Monetary
- ✅ **Diamond Segment Bar Chart** with Customer ID and Score
- ✅ **RFM Table Views & Cluster Statistics**

These visuals help clearly communicate **who your customers are**, how they **behave**, and which ones are **most valuable** for business targeting and retention efforts.

---
## 💡 7. Insights & Business Recommendations

Based on the RFM segmentation and clustering analysis, several actionable insights were discovered to help drive customer engagement and improve business outcomes:

### 🔍 Segment-Specific Insights:

- **💎 Diamond (Cluster 2)**: 
  - These are your top-tier customers — frequent purchasers, recently active, and high spenders.
  - ✅ **Retention Strategy**: Prioritize loyalty programs, VIP benefits, exclusive offers, and personalized experiences to retain them.

- **🥇 Gold (Cluster 3)**: 
  - Valuable customers with moderate frequency and spending.
  - 🔁 **Engagement Strategy**: Upsell and cross-sell through email campaigns, targeted product bundles, or time-limited discounts.

- **🥈 Silver (Cluster 0)**: 
  - Regular customers with low recency — they haven't shopped recently.
  - ⚠️ **Reactivation Strategy**: Use re-engagement campaigns, win-back emails, and time-sensitive promotions.

- **🥉 Bronze (Cluster 1)**: 
  - Infrequent and low-spending customers.
  - 🚀 **Growth Strategy**: Focus on brand awareness, onboarding, or incentives to encourage first-time or repeat purchases.

### 💼 Business Opportunities:

- 🎯 **Personalized Marketing**: Leverage customer scores and segments to drive tailored communication.
- 📉 **Churn Prevention**: Identify at-risk customers by recency drop and trigger retention workflows.
- 📊 **Data-Driven Decisions**: Use the insights from this segmentation to allocate budgets, design campaigns, and inform product strategies.

By implementing these recommendations, businesses can **maximize ROI**, **retain top-value customers**, and **boost overall customer lifetime value (CLV)**.

---
## 📁 8. Repository Structure

The repository contains all relevant files and notebooks used to perform Retail Customer Segmentation using RFM analysis and KMeans clustering.

├── [RFM.ipynb](https://github.com/git-ahsan/RFM-Customer-Segmentation-KMeans/blob/main/RFM.ipynb) # Jupyter Notebook with full RFM & KMeans workflow

├── [Retail Customer Data 2009-10.csv](https://github.com/git-ahsan/RFM-Customer-Segmentation-KMeans/blob/main/Retail%20Customer%20Data%202009-10.zip) # Original dataset used for analysis

├── [README.md](https://github.com/git-ahsan/RFM-Customer-Segmentation-KMeans/blob/main/README.md) # Project documentation

├── [Visualization Charts/](https://github.com/git-ahsan/RFM-Customer-Segmentation-KMeans/tree/main/Visualization%20Charts) # Folder for visualizations (optional: for use in README or reports)

### 📌 Notes:
- The main analysis is performed in `RFM.ipynb`, following a step-by-step methodology from data preprocessing to model training and visualization.
- Ensure required libraries (e.g., pandas, matplotlib, seaborn, scikit-learn, yellowbrick) are installed before running the notebook.
---
## 🧠 9. Key Learnings / What I Learned

This project gave me hands-on experience in applying **real-world customer segmentation** using the RFM model and unsupervised learning. Through this process, I developed the following key skills and insights:

### 🔧 Technical Skills:
- ✅ Performed **data cleaning and preprocessing** on real retail transaction data
- ✅ Applied **RFM feature engineering** to quantify customer behavior
- ✅ Used **KMeans Clustering** and **Elbow Method** to uncover hidden customer segments
- ✅ Normalized and scored customer profiles using **MinMaxScaler** and business logic
- ✅ Visualized complex data using **Matplotlib**, **Seaborn**, and **3D plots**
- ✅ Ranked and labeled customers for practical business decisions

### 🧩 Business Understanding:
- 🎯 Learned how data can uncover **customer value tiers** (e.g., VIPs vs. at-risk customers)
- 📊 Understood how segmentation helps in **targeted marketing and resource allocation**
- 💡 Realized the power of **data-driven decision-making** in improving customer retention

### 💼 Soft Skills:
- 📝 Improved my ability to **document and explain** data science workflows
- 🧠 Strengthened **problem-solving** through exploratory thinking and iterative development

This project not only reinforced my technical foundation but also taught me how to **translate analytics into business insights** — an essential skill for any aspiring data analyst or data scientist.

---

## 🚀 10. Future Work / Improvements

While this project successfully segments customers using RFM analysis and KMeans clustering, there are several opportunities to enhance and expand the work:

- 🔄 **Dynamic RFM Analysis**: Incorporate time-series features to make the segmentation more responsive to changing customer behavior.
- 🧠 **Alternative Clustering Algorithms**: Explore other clustering techniques such as DBSCAN or Hierarchical Clustering for comparison and validation.
- 🎯 **Automated Segment Labeling**: Use rule-based or ML-based techniques to auto-label clusters based on behavior instead of manual assignment.
- 📊 **Dashboard Integration**: Build an interactive dashboard using Power BI, Tableau, or Plotly Dash to allow business users to explore segment insights in real-time.
- 🧪 **AB Testing and Campaign Tracking**: Design marketing strategies for each segment and track real-world performance to refine targeting.
- 📈 **Customer Lifetime Value (CLV)**: Extend the analysis by predicting CLV to make long-term revenue-driven decisions.
- 🌐 **Deployment as a Web App**: Deploy the entire workflow as a user-facing tool where businesses can upload data and view segmentation results directly.

> 🎯 Although currently at a beginner level, I plan to explore areas like **Customer Lifetime Value (CLV)**, **dynamic RFM scoring**, and **deployment techniques** in future projects to enhance segmentation accuracy and business impact.

---
## 📬 Contact & Portfolio Links

**Md. Ahsan Ul Islam**  
🎓 Entry-Level Data Analyst | Skilled in Python, SQL & Power BI | Aspiring Data Scientist with Hands-On ML Project Experience  
🔗 [LinkedIn](https://www.linkedin.com/in/md-ahsan-ul-islam)  
🔗 [GitHub](https://github.com/git-ahsan)

---

## 🏷️ Tags

`#Python` `#DataScience` `#BusinessIntelligence` `#MachineLearning` `#RFM` `#CustomerSegmentation`  
`#ScikitLearn` `#Pandas` `#NumPy` `#KMeans` `#Clustering` `#ElbowMethod`  
`#DataVisualization` `#Matplotlib` `#Seaborn` `#Yellowbrick` `#JupyterNotebook`
