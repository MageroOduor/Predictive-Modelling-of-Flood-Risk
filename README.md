# Predictive-Modelling-of-Flood-Risk

## **Problem Statement**

Rapidly growing cities worldwide face increasing flood risks due to complex interactions between environmental factors (e.g., climate change, monsoon intensity) and urban development (e.g., urbanization, inadequate drainage systems). However, real-world flood data is often scarce or inconsistent, making accurate risk prediction a significant challenge. This project addresses the critical need for robust flood risk prediction by performing a case study using a synthetic, multi-domain environmental indicator dataset. 

## **General Objective**

To develop and evaluate machine learning models that can effectively identify high-risk areas and influential factors, providing a framework that can inform disaster preparedness, urban planning, and resource allocation strategies, ultimately aiming to mitigate the socio-economic impacts of flooding, even when real-world data limitations exist.

## **Specific Objectives**

1. Conduct comprehensive Exploratory Data Analysis (EDA) to understand feature distributions, identify correlations, and detect potential anomalies in the dataset;
2. Perform feature engineering and preprocessing (e.g., scaling) to optimize the data for machine learning models;
3. Build, evaluate, and interpret multiple machine learning models (Linear Regression, Random Forest, and XGBoost) to predict flood probability using RMSE, and identify the most influential predictors of flood risk; and
4. Deploy the best-performing model using Streamlit to develop an interactive interface for real-time flood probability prediction.

## **Significance of the Project**

This project demonstrates the practical value of machine learning in addressing real-world flood risks by: 
1. Providing a reliable, data-driven approach to predicting flood risk in rapidly growing urban areas;
2. Supporting better urban planning by identifying high-risk areas and guiding infrastructure decisions;
3. Enhancing disaster preparedness through early risk detection and improved response planning; and
4. Contributing to reducing socio-economic losses by enabling targeted and efficient resource allocation.

## Bias and Fairness

Bias in machine learning models, especially those used for critical predictions like flood risk, can lead to severe and inequitable outcomes. While this is a synthetic dataset, it's crucial to consider potential sources of bias and their implications for fairness if such a model were deployed in real-world scenarios.

#### Potential Sources of Bias in the Dataset:
1.  **PopulationScore:** This feature likely represents population density or vulnerability. If the synthetic generation implicitly links lower `PopulationScore` (e.g., less dense or wealthier areas) to lower flood probability, or higher scores (e.g., denser, potentially poorer areas) to higher flood probability, it could perpetuate or amplify existing societal inequalities. In reality, vulnerable populations often live in high-risk areas with inadequate infrastructure.
2.  **Urbanization:** Urbanization is a complex process. If the dataset's `Urbanization` metric disproportionately penalizes certain types of urban development or implicitly favors others, the model could recommend policies that disadvantage specific communities or types of urban growth.
3.  **PoliticalFactors:** This feature is highly abstract and fraught with potential for bias. If `PoliticalFactors` are correlated with lower flood probability, it might imply that areas with certain political characteristics (which could be proxies for socioeconomic status, power, or governance effectiveness) receive preferential treatment in flood management or are inherently less at risk in the synthetic world. In a real-world context, such a feature could be a proxy for corruption, political will, or historical underinvestment, leading to unfair risk assessments for marginalized communities.

#### Implications of Synthetic Data on Observed Biases:
Being a synthetic dataset, any observed biases are a direct result of the data generation process. This could be a strength, allowing for controlled study, or a limitation, as it might not fully capture the nuanced and often hidden biases present in real-world data. Real-world biases often stem from historical discrimination, data collection methods, and societal inequalities that are hard to replicate synthetically. Therefore, while this synthetic data can illustrate *types* of bias, the *magnitude* and *nature* of real-world bias could be significantly different.

#### Ethical Implications and Fairness Issues:
If a biased model for flood risk prediction were deployed, it could have severe ethical and fairness implications:
*   **Resource Allocation:** A model biased against certain `PopulationScore` or `Urbanization` levels could lead to under-investment in flood preparedness or infrastructure in specific communities, increasing their vulnerability.
*   **Insurance and Lending:** Biased predictions could result in higher insurance premiums or denial of loans for property development in certain areas, exacerbating economic disparities.
*   **Disaster Preparedness:** Emergency response planning based on biased risk assessments could lead to inadequate early warning systems or evacuation routes for disproportionately affected communities.
*   **Environmental Justice:** Flood risk is often a matter of environmental justice, where marginalized groups are more exposed to hazards. A biased model could deepen these existing injustices.

#### Conclusion on Bias Mitigation:
This analysis primarily focuses on predictive performance and interpretability. Comprehensive bias detection and mitigation strategies (e.g., re-sampling, re-weighting, adversarial debiasing, or fairness-aware algorithms) are not explicitly performed in this project. However, the initial EDA and feature importance analysis serve as critical first steps in identifying features that *could* contribute to bias. For real-world applications, a dedicated and thorough bias audit, involving domain experts and affected communities, would be an indispensable next step to ensure ethical and fair deployment of flood risk prediction models.

## **Methodology**

## **Data Source and Understanding**

The dataset used in this project was sourced from Kaggle https://www.kaggle.com/code/marianadeem755/forecasting-flood-unlocking-regression-advancement?scriptVersionId=179074936&cellId=3. The data was pre-split into training and testing sets, with the training dataset containing 1,117,957 records and 22 variables, and the testing dataset containing 745,305 records and 21 variables.

The dataset is entirely numerical and captures multiple environmental, infrastructural, and socio-economic factors influencing flood risk. Key predictor variables include *Monsoon Intensity, Topography Drainage, River Management, Deforestation, Urbanization, Climate Change, Drainage Systems, Coastal Vulnerability,* and *Wetland Loss*, among others. The target variable is *Flood Probability*, which represents the likelihood of flooding on a continuous scale.

Initial data checks confirmed that there were **no missing values or duplicate records**, making the dataset suitable for direct modeling.

## **Exploratory Data Analysis(EDA) and Preprocessing**

Exploratory Data Analysis (EDA) was conducted to understand the distribution, variability, and relationships among variables. Summary statistics such as mean, minimum, maximum, and standard deviation were computed, while visualizations including histograms were used to assess data distribution and detect anomalies.

The analysis showed that the dataset is relatively well-distributed with no significant outliers. Since all variables are numerical, preprocessing mainly involved **feature scaling using StandardScaler** to standardize the data and improve model performance.

The dataset was then split into training and testing sets to enable proper model evaluation.

### **Feature Relationships with Flood Probability**

We explored how individual features relate to the target variable, FloodProbability by visualizing distributions for all numerical features in the train DataFrame, excluding the id and FloodProbability columns, to understand their individual distributions. We ensured the plots are well-organized using subplots and created a correlation matrix by displaying it as a heatmap to identify relationships between variables.

<img width="1614" height="1490" alt="image" src="https://github.com/user-attachments/assets/0a99c118-6ccf-44f7-bb38-cb2845c5ab04" />


<img width="1672" height="1664" alt="image" src="https://github.com/user-attachments/assets/2bd73cf2-2408-4ea6-846a-62f9acc6a481" />

<img width="1070" height="1004" alt="image" src="https://github.com/user-attachments/assets/f4de841b-d6b0-4b6c-91a4-8b857fd3e466" />

## Summary of EDA

#### Feature Distributions (Histograms):
Most of the features are fairly evenly distributed between 1 and 10, with no extreme values, missing data issues, or unusual patterns. This means there was no need for additional preprocessing steps such as outlier removal, imputation, or data transformation before modeling.

#### Correlation Matrix Analysis

1.  **Target Variable (`FloodProbability`) Correlations:**
    *   `FloodProbability` shows moderate positive correlations with several original features, notably `MonsoonIntensity` (0.19), `TopographyDrainage` (0.18), `ClimateChange` (0.19), and `DeterioratingInfrastructure` (0.19). This confirms the expected influence of these factors on flood risk.
    *   The engineered features, as seen in the second heatmap, show stronger correlations with `FloodProbability`, notably `EnvironmentalIndex`(**0.58**), `UrbanIndex`(**0.59**), `Env_Urban_Interaction`(**0.64), `EnvironmentalIndex_sq` (**0.56**),  `UrbanIndex_sq`(**0.57**)and `Env_Urban_Interaction_sq`(**0.62**)

    These values demonstrate that combining related factors into indices effectively captures and amplifies their collective impact on flood risk, validating our feature engineering.

2.  **Inter-Feature Correlations (Multicollinearity):**
    *   Many individual environmental features (e.g., `MonsoonIntensity`, `TopographyDrainage`, `Deforestation`, `ClimateChange`, `Siltation`, `Watersheds`, `WetlandLoss`, `Landslides`) showed low to moderate positive correlations among themselves. This is expected as environmental factors can be interconnected.
    *   Similarly, urban features (e.g., `Urbanization`, `DamsQuality`, `AgriculturalPractices`, `Encroachments`, `IneffectiveDisasterPreparedness`, `DrainageSystems`, `CoastalVulnerability`, `DeterioratingInfrastructure`, `PopulationScore`, `InadequatePlanning`, `PoliticalFactors`) also exhibit low to moderate positive correlations with each other.

    * **The engineered indices, `EnvironmentalIndex` and `UrbanIndex`, exhibited a weak negative correlation of-0.11, indicating a largely independent and slightly inverse relationship between overall environmental and urban conditions.** In contrast, the `Env_Urban_Interaction` term (their product) shows a strong positive correlation with both `EnvironmentalIndex` (**0.71**) and `UrbanIndex` (**0.61**), reflecting its dependence on both contributing variables. Additionally, the squared features demonstrate extremely high correlations with their respective linear forms (e.g., `EnvironmentalIndex` and `EnvironmentalIndex_sq` at ~0.99), indicating the presence of multicollinearity among the engineered variables. While this level of multicollinearity may affect the stability of coefficient estimates in linear models, it i

**Key Takeaways for Modeling**
*   The  engineered features demonstrated strong predictive value for FloodProbability, with indices such as EnvironmentalIndex, UrbanIndex, and Env_Urban_Interaction showing notably higher correlations than individual raw features. This confirms that aggregating related environmental and urban factors effectively captures deeper patterns influencing flood risk.
*   Compared to the moderate correlations observed in the original variables, the engineered features consistently showed stronger relationships with the target variable, reinforcing the effectiveness of the multi-domain feature engineering in improving signal strength for model learning.
*   Although individual environmental and urban indicators exhibited low to moderate inter-correlations, the engineered indices introduced a higher degree of multicollinearity, particularly among interaction and polynomial terms. This is an expected outcome of feature transformation and should be considered during model selection and interpretation.

*   Tree-based models such as Random Forest are well-suited for this structure due to their robustness to multicollinearity, while linear models may require additional regularization or feature selection to ensure stable coefficient interpretation.

#### Data Preprocessing and Feature Engineering

This step focused on preparing the dataset for modeling by improving feature representation and ensuring consistency in feature scales to enhance model performance.

### 1. Data Preparation
We:
1. Removed non-informative columns such as identifiers.
2. Confirmed that all variables were numerical, so no encoding was required.
3. Verified data quality (no missing values or duplicates).
   
### 2. Feature Engineering

Based on insights from EDA, where flood risk was influenced by combined environmental and urban effects we:

1. Created an Environmental Index by aggregating key environmental variables.
2. Created an Urban Index by combining urban-related variables.
3. Developed interaction features to capture combined environmental–urban effects.

These transformations were introduced to help models learn joint effects rather than isolated feature impacts.

### 3. Feature Scaling

Since the dataset contains numerical features with varying scales, we applied StandardScaler to normalize all variables. This ensured equal contribution of features during training and improved model stability and performance, particularly for regression models and hyperparameter tuning.

### **Model Development**

A stepwise modeling approach was adopted to ensure robustness and comparability of results.

First, **Linear Regression** was used as a baseline model to establish a reference performance for prediction accuracy.

Subsequently, more advanced machine learning models were implemented, including:

* **Random Forest Regressor**, to capture nonlinear relationships and feature interactions through ensemble learning
* **XGBoost Regressor**, to enhance predictive performance using gradient boosting and sequential learning

This progression allowed for comparison between simple and complex models to identify the most effective approach for flood prediction.

### **Model Evaluation**

Model performance was evaluated using **Root Mean Squared Error (RMSE)** as the primary metric, as it measures the average magnitude of prediction errors and is well-suited for regression problems.

Lower RMSE values indicate better model performance. The models were compared based on their RMSE scores on the test dataset, and the best-performing model was selected for final predictions and analysis.

### **Model Deployment**

To enhance usability and practical application, the final model was deployed using Streamlit, an interactive web application framework.

A user-friendly interface was developed to allow users to input relevant environmental and infrastructural variables and obtain real-time flood probability predictions. This deployment demonstrates the practical applicability of the model and enables non-technical users, such as planners and decision-makers, to easily interact with the system.
