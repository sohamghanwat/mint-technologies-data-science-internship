# Data Science Internship Project - Mint Technologies

## About
This repo holds my work for the Data Science internship at Mint Technologies (Cohort-24).

I use one dataset for the whole project: **House Sales in King County, USA** (Kaggle).
Dataset link: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

## Project Structure
```
data-science-internship/
├── data/
│   ├── kc_house_data.csv
│   └── engineered_data.csv
├── notebooks/
│   ├── 01_exploration_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_advanced_models.ipynb
└── README.md
```

## Milestone 1: Exploration & Cleaning
In this milestone, I explore and clean the housing data.

What the notebook does:
- Loads the CSV file into a Pandas DataFrame.
- Shows the shape, data types, summary numbers and missing values.
- Handles missing values and explains when to fill or drop.
- Makes 3 charts: a histogram of `price`, a correlation heatmap and a scatter plot of the top 2 correlated features.

## Milestone 2: Feature Engineering
In this milestone, I change the data into a form a model can use.

What the notebook does:
- Finds the categorical columns and changes them to numbers (Label Encoding and One-Hot Encoding).
- Scales the number columns with `StandardScaler`.
- Creates new features: `house_age`, `renovation_status`, `season` and `total_rooms`. The notebook explains why each one helps.
- Saves the final data to `data/engineered_data.csv`.

## Milestone 3: Baseline Model
In this milestone, I train my first model.

What the notebook does:
- Splits the data into features (X) and target (y), then into 80% training and 20% testing sets.
- Trains a Linear Regression model, because `price` is a number.
- Checks the model with MSE, RMSE and R2 on the test set.
- Compares the model with just guessing the average price and explains if the baseline is good.

## Milestone 4: Model Comparison
In this milestone, I try stronger models and compare them with the baseline.

What the notebook does:
- Trains Random Forest and Gradient Boosting models.
- Tunes both with `GridSearchCV`.
- Shows a comparison table with the train score, test score and best settings of each model.
- Checks the gap between train and test scores to look for overfitting.
- Gives a written conclusion on the best model.

## How to Run
1. Install the libraries:
   `pip install pandas numpy matplotlib seaborn jupyter`
2. Download `kc_house_data.csv` from the Kaggle link above and put it in the `data/` folder.
3. Start Jupyter:
   `jupyter notebook`
4. Open `notebooks/01_exploration_cleaning.ipynb` and run all cells.
5. Install scikit-learn:
   `pip install scikit-learn`
6. Open `notebooks/02_feature_engineering.ipynb` and run all cells. This creates `data/engineered_data.csv`.
7. Open `notebooks/03_baseline_model.ipynb` and run all cells.
8. Open `notebooks/04_advanced_models.ipynb` and run all cells. This one can take a few minutes because of the grid search.

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Jupyter Notebook
