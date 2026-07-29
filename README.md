<img width="148" height="148" alt="waze" src="https://github.com/user-attachments/assets/b0855f29-fd86-4f58-b8a1-1a412a702683" />

# Case Study: Driving Retention — Waze User Churn Analysis & Prediction
## Project Objective: 
Analyze monthly user journey data to identify the behavioral drivers behind user churn and build predictive machine learning models to help Waze's marketing and product teams proactively run targeted retention campaigns.  
#### Tools & Libraries: Python | Pandas | NumPy | Seaborn | Plotly | Scikit-Learn | XGBoost  
## 📌 Executive Summary & Core Insights
The Waze dataset consists of synthetic user activity logs (~15,000 users) with an overall churn rate of 17.7% (82.3% retention).

<img width="335" height="222" alt="waze user proportion" src="https://github.com/user-attachments/assets/33aea6d2-303c-483f-bf43-4082acd1d6c3" />

Through **Exploratory Data Analysis (EDA)**, several highly unexpected behavioral patterns were uncovered:

The "Long-Haul Burnout" Paradox: Churned users didn't leave because they weren't using the app; they actually drove longer distances and more hours than retained users. The median churned user drove ~200 km more and 2.5 hours more per month than retained users. 

<img width="335" height="222" alt="churn_distance" src="https://github.com/user-attachments/assets/e1e9ab02-9fba-40bf-8c77-598f9e822f5a" />


Frequency Beats Distance: The single best predictor of retention was activity frequency, not total usage. Users who opened and drove with the app 30 days in a month had a 0% churn rate. 

<img width="335" height="222" alt="churn_driving days" src="https://github.com/user-attachments/assets/1c46286b-1e57-4a86-8b89-de4adc2a0ccc" />

### Predictive Modeling Insights


**The Insight**: The data shows that user churn at Waze cannot be easily predicted purely by volume of usage (kilometers or days driven).

**The Evidence**: Even when optimizing the decision threshold down to an aggressive 0.194, the model could still only capture 50% of the true churn events, proving that the remaining churned users behave identically to active users within our current feature set.


## Schema:

| Column name | Type | Description |
| :--- | :--- | :--- |
| ID | int | A sequential numbered index |
| label | obj | Binary target variable (“retained” vs “churned”) for if a user has churned anytime during the course of the month |
| sessions | int | The number of occurrence of a user opening the app during the month |
| drives | int | An occurrence of driving at least 1 km during the month |
| device | obj | The type of device a user starts a session with |
| total_sessions | float | A model estimate of the total number of sessions since a user has onboarded |
| n_days_after_onboarding | int | The number of days since a user signed up for the app |
| total_navigations_fav1 | int | Total navigations since onboarding to the user’s favorite place 1 |
| total_navigations_fav2 | int | Total navigations since onboarding to the user’s favorite place 2 |
| driven_km_drives | float | Total kilometers driven during the month |
| duration_minutes_drives | float | Total duration driven in minutes during the month |
| activity_days | int | Number of days the user opens the app during the month |
| driving_days | int | Number of days the user drives (at least 1 km) during the month |


## 🛠️ The Data Analytics Workflow
### 1. Data Cleaning & Feature Engineering

#### Missing Values:

No missing values were found.

#### Outliers:

<img width="531" height="169" alt="outlier" src="https://github.com/user-attachments/assets/d40cbfea-6488-44b2-8ec0-99f0e3e4f5fa" />

<img width="535" height="195" alt="histogram" src="https://github.com/user-attachments/assets/4bdba2e3-6bec-4707-8891-3384d851e16f" />


The box plots indicated that many of these variables have outliers which do not seem to be data entry errors; they are present because of the right-skewed distributions.

To impute, outlying data with more reasonable value, is to set a threshold based on a percentile of the distribution.

#### Validation:
Drive-days cannot have more than activity-days. None of the samples in this data violate this rule.

<img width="341" height="202" alt="scatterplot_user days" src="https://github.com/user-attachments/assets/517a0065-8b4d-4cb0-95cf-94ba3bd33f61" />

#### Feature Engineering: 
Engineered critical new ratios to expose user behaviors that raw data hid:

km_per_driving_day (revealed the long-distance correlation)  
percent_sessions_to_drives (user intent vs. execution)
professional_driver flag (users logging $>60$ drives/month—who turned out to have a drastically lower churn rate of 7.6%).  
### 2. Hypothesis Testing for Device Type
Hypotheses:

𝐻0: There is no difference in average number of drives between drivers who use iPhone devices and drivers who use Androids.

𝐻𝐴: There is a difference in average number of drives between drivers who use iPhone devices and drivers who use Androids.

Next, choose 5% as the significance level and proceed with a two-sample t-test.



#### pvalue = 0.159

Since the p-value is larger than the chosen significance level (5%), you fail to reject the null hypothesis. You conclude that there is not a statistically significant difference in the average number of drives between drivers who use iPhones and drivers who use Androids.

## Machine Learning Models

### Logistic Regression Model
#### Assumptions
1. Independent observations (This refers to how the data was collected.)
2. No extreme outliers.
3. Little to no multicollinearity among X predictors.

<img width="449" height="335" alt="heatmap_reg model" src="https://github.com/user-attachments/assets/d0bc0140-1e11-4478-bda7-8574c0a89106" />

4. Linear relationship between X and the logit of y.

<img width="382" height="219" alt="regplot" src="https://github.com/user-attachments/assets/139d102a-29aa-4151-a111-f1607c468c5a" />

#### Results with a Confusion Matrix

 <img width="354" height="232" alt="Logistic_confusion matrix" src="https://github.com/user-attachments/assets/51ed86c3-4724-4aea-ba55-d2ec791a6c72" />

#### Classification Report

| Class / Metric | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Retained** | 0.83 | 0.98 | 0.90 | 2941 |
| **Churned** | 0.54 | 0.09 | 0.16 | 634 |
| | | | | |
| **Accuracy** | | | **0.82** | **3575** |
| **Macro Avg** | 0.69 | 0.54 | 0.53 | 3575 |
| **Weighted Avg** | 0.78 | 0.82 | 0.77 | 3575 |

The model has decent precision but very low recall, which means that it makes a lot of false negative predictions and fails to capture users who will churn.

#### New Features for Modelling

percent_sessions_in_last_month,
total_sessions_per_day,
km_per_hour,
km_per_drive,
percent_of_sessions_to_favorite

### Random Forest Model
| Model | Precision | Accuracy | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- |
| **RF cv** | 0.469455 | 0.820026 | 0.111682 | 0.180308 |

#### Findings:

Asside from the accuracy, the scores aren't that good. However, recall that when we built the logistic regression model the recall was ~0.09, which means that this model has 33% better recall and about the same accuracy, and it was trained on less data.

### XGBoost

| Model | Precision | Accuracy | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- |
| **RF cv** | 0.469455 | 0.820026 | 0.111682 | 0.180308 |
| **XGB cv** | 0.404109 | 0.807903 | 0.174116 | 0.243323 |



#### Model selection

Now, use the best random forest model and the best XGBoost model to predict on the validation data. Whichever performs better will be selected as the champion model.

| Model | Precision | Accuracy | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- |
| **RF cv** | 0.469455 | 0.820026 | 0.111682 | 0.180308 |
| **XGB cv** | 0.404109 | 0.807903 | 0.174116 | 0.243323 |
| **RF val** | 0.433071 | 0.816783 | 0.108481 | 0.173502 |
| **XGB val** | 0.387097 | 0.805594 | 0.165680 | 0.232044 |

### Champion model to predict on test data
Now, use the champion model to predict on the test dataset. This is to give a final indication of how you should expect the model to perform on new future data, should you decide to use the model.

| Index | Model | Precision | Accuracy | Recall | F1 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | **RF cv** | 0.469455 | 0.820026 | 0.111682 | 0.180308 |
| 1 | **XGB cv** | 0.404109 | 0.807903 | 0.174116 | 0.243323 |
| 2 | **RF val** | 0.433071 | 0.816783 | 0.108481 | 0.173502 |
| 3 | **XGB val** | 0.387097 | 0.805594 | 0.165680 | 0.232044 |
| 4 | **XGB test** | 0.404255 | 0.806993 | 0.187377 | 0.256065 |

#### XG Boost Confusion matrix

<img width="327" height="207" alt="xgboost_confusion matrix" src="https://github.com/user-attachments/assets/46c6a3ef-f239-4a64-a87e-5785d1306eb3" />



The model predicted three times as many false negatives than it did false positives, and it correctly identified only 16.6% of the users who actually churned.

#### Feature importance

<img width="539" height="312" alt="feature_imp_xgb" src="https://github.com/user-attachments/assets/eccab794-e411-4716-a682-9635f0ec1c7c" />

#### Identify an optimal decision threshold
The default decision threshold for most implementations of classification algorithms—including scikit-learn's—is 0.5. With imbalanced datasets where the response class is a minority, this threshold might not be ideal. You learned that a precision-recall curve can help to visualize the trade-off between your model's precision and recall.

Here's the precision-recall curve for the XGBoost champion model on the test data.


<img width="385" height="335" alt="precision_recall_curve_xgb" src="https://github.com/user-attachments/assets/841a8083-c806-4b5a-b6cc-b1181f21e0d6" />

In the case of this Waze project, a false positive could just mean that a user who will not actually churn gets an email and a  notification on their phone. It's very low risk.



With lower threshold of **0.4**:

| Index | Model | Precision | Recall | F1 | Accuracy |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | XGB, threshold = 0.4 | 0.385350 | 0.238659 | 0.294762 | 0.797552 |

If the model successfully identified half of the people who will actually churn, even if it means that when the model says someone will churn, it's only correct about 30% of the time.

### Threshold to yield the result

| Index | Model | Precision | Recall | F1 | Accuracy |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | XGB, threshold = 0.194 | 0.329212 | 0.420118 | 0.369151 | 0.745455 |

### The Recommendation: 
To break past this 50% wall, the organization needs to collect behavioral sentiment data (e.g., user reports filed, app crashes experienced, or navigation layout changes) rather than just raw mileage metrics.
