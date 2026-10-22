# House Price Predictor

A Data Science internship project at Mint Technologies (Cohort-24). It predicts house prices and shows the predictions in a Streamlit web app.

## Problem Statement
This project predicts the **sale price of a house** (in US dollars) from its details, such as size, grade, location and year built.

The target is a number, so this is a **regression** problem.

## Dataset
The data is **House Sales in King County, USA** from Kaggle.

- Link: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
- It has about 21,600 house sales in King County (Seattle area), from May 2014 to May 2015.
- Each row is one house. Each column is a detail of the house (for example `sqft_living`, `grade`, `zipcode`). The column `price` is the target.

## Approach
1. **Cleaning** (notebook 01): I checked the shape, data types and missing values. I made charts of the price, the correlations and the top 2 features.
2. **Features** (notebook 02):
   - I made 4 new features: `house_age`, `renovation_status`, `season` and `total_rooms`.
   - I changed text columns to numbers with Label Encoding and One-Hot Encoding.
   - I scaled the number columns with `StandardScaler`.
   - I did not use `price_per_sqft`, because it is made from the target and would leak the answer.
3. **Modeling** (notebooks 03 and 04):
   - I trained a Linear Regression baseline.
   - I trained Random Forest and Gradient Boosting and tuned them with `GridSearchCV`.
   - I compared all models on the train and test sets and checked for overfitting.
4. **Deployment**: I saved the final model with `joblib` and built the Streamlit app in `app.py`.

## Results
**Winning model:** `<write the model name here>`

| Model | Train Score (R2) | Test Score (R2) |
|-------|------------------|-----------------|
| Baseline (Linear Regression) | `<fill in>` | `<fill in>` |
| Random Forest | `<fill in>` | `<fill in>` |
| Gradient Boosting | `<fill in>` | `<fill in>` |

**Test RMSE of the winning model:** `<fill in>` dollars

**Why it won:** `<write one or two sentences from notebook 04>`

## How to Run
1. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
2. Download `kc_house_data.csv` from the Kaggle link above and put it in the `data/` folder.
3. Run notebooks `02_feature_engineering.ipynb` and `04_advanced_models.ipynb` (run `03` too if you want the baseline). They create `preprocessing.joblib` and `model.joblib` in the main folder. These two files must be in the same folder as `app.py`.
4. Start the app:
   ```
   streamlit run app.py
   ```
5. Open the link that Streamlit shows (usually http://localhost:8501). Upload a CSV file. You can use `data/kc_house_data.csv`. The app shows the predicted price for each house.
