# House Price Prediction — King County, USA
A machine learning project that predicts residential house prices in King County, Washington using regression models including MLP, Random Forest, and XGBoost.
---
## Overview
This project builds and compares multiple supervised regression models to predict house sale prices based on property features such as size, location, condition, and age. 
The dataset used is the publicly available **King County House Sales** dataset.
---
## Dataset

| Property | Detail |
|---|---|
| **Source** | King County House Sales (`kc_house_data.csv`) |
| **Records** | ~21,613 entries |
| **Features** | 21 columns (bedrooms, bathrooms, sqft_living, floors, waterfront, grade, zipcode, lat/long, etc.) |
| **Target** | `price` (sale price in USD) |

---

## Tech Stack
- **Language:** Python 3
- **Notebook:** Jupyter Notebook
- **Libraries:**
  - `pandas`, `numpy` — data handling
  - `matplotlib`, `seaborn`, `plotly` — visualization
  - `scipy` — statistical analysis
  - `scikit-learn` — preprocessing, model training, evaluation
  - `xgboost` — gradient boosting
---

## Exploratory Data Analysis (EDA)

The EDA phase explores relationships between features and the target variable:
- **Correlation heatmap** — identified high correlation (0.88) between `sqft_living` and `sqft_above`; `sqft_above` was dropped
- **Boxplots** — price distribution across number of floors, bedrooms, and bathrooms
- **Violin plot** — price vs. number of views
- **Bar plots** — price by waterfront presence and zipcode
- **Scatter plot** — price vs. sqft_living coloured by floors
- **Interactive map** — geographic distribution of houses using Plotly + OpenStreetMap

## Feature Engineering & Preprocessing

| Step | Description |
| **Outlier Removal** | Applied 3-sigma rule to remove price outliers (`zscore < 3`) |
| **Bedroom filter** | Dropped entries with more than 10 bedrooms (treated as outliers) |
| **Year extraction** | Extracted sale year from the `date` column |
| **House age** | New feature: `age = year - yr_built` |
| **Renovation flag** | Binary feature: `renovated = 1` if `yr_renovated > 0` |
| **Renovation age** | New feature: years since last renovation |
| **Age bins (OHE)** | Age bucketed into 13 bins (`<0` to `111-120`) and one-hot encoded |
| **Renovation age bins (OHE)** | Renovation age bucketed into 9 bins and one-hot encoded |
| **Categorical encoding** | `view`, `condition`, and `zipcode` one-hot encoded via `pd.get_dummies` |
| **Dropped columns** | `id`, `sqft_above`, `year`, `yr_built`, `yr_renovated` |
| **Feature scaling** | `StandardScaler` applied to all features |
| **Train/Test Split** | 80% training, 20% testing (`random_state=1`) |

## Models
### 1. MLP Regressor (Neural Network)
```
Hidden Layers : (80, 100)
Activation    : ReLU
Solver        : Adam
Learning Rate : 0.01 (constant)
Max Iterations: 2000
Early Stopping: Enabled
```
### 2. Random Forest Regressor
```
n_estimators : 300
max_depth    : 30
max_features : None
random_state : 0
```
### 3. XGBoost Regressor
```
n_estimators  : 300
learning_rate : 0.10
booster       : gbtree
gamma         : 2
max_depth     : 10
```

## Evaluation Metrics
## All models are evaluated using:
| Metric | Description |
| **R² Score** | Proportion of variance explained by the model |
| **RMSE** | Root Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **Cross-Validation** | 10-fold CV score |
Model performance is visualised via a bar chart comparing R² scores across all three algorithms, and a histogram comparing predicted vs. actual house prices (XGBoost). 
Feature importance for XGBoost is also plotted (top 20 features).

## Getting Started
### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost scipy
```
### Run the Notebook
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
jupyter notebook HousePricePrediction.ipynb
```
> **Note:** Update the dataset path in the notebook to point to your local copy of `kc_house_data.csv`.

## Project Structure
```
house-price-prediction/
│
├── HousePricePrediction.ipynb   # Main Jupyter notebook
├── kc_house_data.csv            # Dataset (download separately)
└── README.md                    # Project documentation
```
## License
This project is for academic/educational purposes.
