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

## **Methodology**

## **Data Source and Understanding**

The dataset used in this project was sourced from Kaggle under the “Forecasting Flood Levels: Unlocking Regression Advancements” competition. The data was pre-split into training and testing sets, with the training dataset containing 1,117,957 records and 22 variables, and the testing dataset containing 745,305 records and 21 variables.

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

## Summary of EDA

#### Feature Distributions (Histograms):
Most of the original environmental and urban indicator features showed a relatively uniform or slightly skewed distribution across their discrete values (typically 1-10). This indicates that the dataset is synthetic and likely generated to cover a range of scenarios for each indicator. There are no immediate outliers or extremely sparse categories that would require special handling, aside from the general discrete nature of these features.

#### Correlation Matrix Analysis:
1.  **Target Variable (`FloodProbability`) Correlations:**
    *   `FloodProbability` showed moderate positive correlations with several features, notably `MonsoonIntensity` (0.19), `TopographyDrainage` (0.18), `ClimateChange` (0.19), and `DeterioratingInfrastructure` (0.19). This confirms the expected influence of these factors on flood risk.
    *   The engineered features `EnvironmentalIndex`, `UrbanIndex`, and `Env_Urban_Interaction` showed stronger correlations with `FloodProbability` (0.42, 0.44, and 0.48 respectively) compared to individual raw features. This suggests that combining these related factors into indices effectively captures their collective impact on flood risk, validating our feature engineering approach.

2.  **Inter-Feature Correlations (Multicollinearity):**
    *   Many individual environmental features (e.g., `MonsoonIntensity`, `TopographyDrainage`, `Deforestation`, `ClimateChange`, `Siltation`, `Watersheds`, `WetlandLoss`, `Landslides`) showed low to moderate positive correlations among themselves. This is expected as environmental factors can be interconnected.
    *   Similarly, urban features (e.g., `Urbanization`, `DamsQuality`, `AgriculturalPractices`, `Encroachments`, `IneffectiveDisasterPreparedness`, `DrainageSystems`, `CoastalVulnerability`, `DeterioratingInfrastructure`, `PopulationScore`, `InadequatePlanning`, `PoliticalFactors`) also exhibited low to moderate positive correlations with each other.
    *   **The engineered indices themselves, `EnvironmentalIndex` and `UrbanIndex`, showed a strong positive correlation (0.75).** This indicates significant overlap or interdependence between overall environmental and urban conditions in influencing flood risk, which is captured and amplified by the `Env_Urban_Interaction` term. This strong correlation between the two main indices might lead to multicollinearity issues in simple linear models but is often handled well by tree-based models like Random Forest. The `Env_Urban_Interaction` term, as expected, also has strong correlations with both `EnvironmentalIndex` (0.97) and `UrbanIndex` (0.95), further reinforcing its role as a combined effect.

**Key Takeaways for Modeling:**
*   The engineered features, especially `Env_Urban_Interaction`, appeared to be highly predictive of `FloodProbability`, which is a positive outcome of our feature engineering step.
*   The moderate correlations between individual features and the target, and the stronger correlations of the engineered indices, supported the multi-domain approach to flood risk prediction.
*   The presence of multicollinearity among the engineered indices was noted, especially if using models sensitive to it (like Linear Regression without regularization). Tree-based models are generally more robust to this. This EDA confirms that the engineered features are meaningful and capture significant variance in the target variable.

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
