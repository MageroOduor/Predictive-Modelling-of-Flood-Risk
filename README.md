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

### **Data Source and Understanding**

The dataset used in this project was sourced from Kaggle under the “Forecasting Flood Levels: Unlocking Regression Advancements” competition. The data was pre-split into training and testing sets, with the training dataset containing 1,117,957 records and 22 variables, and the testing dataset containing 745,305 records and 21 variables.

The dataset is entirely numerical and captures multiple environmental, infrastructural, and socio-economic factors influencing flood risk. Key predictor variables include *Monsoon Intensity, Topography Drainage, River Management, Deforestation, Urbanization, Climate Change, Drainage Systems, Coastal Vulnerability,* and *Wetland Loss*, among others. The target variable is *Flood Probability*, which represents the likelihood of flooding on a continuous scale.

Initial data checks confirmed that there were **no missing values or duplicate records**, making the dataset suitable for direct modeling.

### **Exploratory Data Analysis and Preprocessing**

Exploratory Data Analysis (EDA) was conducted to understand the distribution, variability, and relationships among variables. Summary statistics such as mean, minimum, maximum, and standard deviation were computed, while visualizations including histograms were used to assess data distribution and detect anomalies.

The analysis showed that the dataset is relatively well-distributed with no significant outliers. Since all variables are numerical, preprocessing mainly involved **feature scaling using StandardScaler** to standardize the data and improve model performance.

The dataset was then split into training and testing sets to enable proper model evaluation.

<img width="1614" height="1490" alt="image" src="https://github.com/user-attachments/assets/0a99c118-6ccf-44f7-bb38-cb2845c5ab04" />


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

### **Overall Model Building Process, Key Findings**

### **Overall Model Building Process:**
We embarked on a comprehensive machine learning analysis for flood risk prediction, following key data science principles:
1.  **Exploratory Data Analysis (EDA):** We loaded the synthetic dataset, inspected its shape and missing values, and visualized feature distributions and correlations. The EDA revealed that most original features had discrete values and moderate correlations with flood probability. Crucially, it highlighted the potential of engineered features to capture more complex relationships.
2.  **Feature Engineering:** Building on EDA insights, we created `EnvironmentalIndex`, `UrbanIndex`, and an `Env_Urban_Interaction` term. Further refinement involved adding polynomial (square) terms for these engineered features to capture non-linearities.
3.  **Data Preprocessing:** All features were scaled using `StandardScaler` to ensure fair contribution during model training, especially for models sensitive to feature scales.
4.  **Addressing Bias and Fairness:** We performed a conceptual analysis of potential biases in the synthetic dataset, particularly concerning features like `PopulationScore`, `Urbanization`, and `PoliticalFactors`, and discussed their ethical implications for real-world deployment.
5.  **Model Building and Evaluation:**
    *   **Linear Regression (Baseline):** A simple Linear Regression model was trained with the engineered and scaled features. It provided a baseline RMSE and R-squared, showing a decent initial fit, and its coefficients offered insights into feature directionality.
    *   **Random Forest Regressor (Improvement):** A Random Forest model was implemented to capture non-linear relationships, demonstrating improved performance over Linear Regression. Feature importance from Random Forest highlighted the `Env_Urban_Interaction` term as overwhelmingly important.
    *   **XGBoost Regressor (Advanced Model):** An XGBoost model was trained, further improving performance metrics (lower RMSE, higher R-squared) compared to both Linear Regression and Random Forest. Similar to Random Forest, XGBoost also identified `Env_Urban_Interaction` as the most critical feature, with `EnvironmentalIndex` and `UrbanIndex` having secondary importance. The polynomial terms, while intended to capture more complexity, did not show significant direct importance in the tree-based models, likely due to the models' inherent ability to find such interactions.

### Key Findings:
*   **Engineered Features are Crucial:** The composite `EnvironmentalIndex`, `UrbanIndex`, and especially their interaction term `Env_Urban_Interaction`, consistently proved to be the most powerful predictors of `FloodProbability`. They significantly improved model performance compared to using individual raw features.
*   **Non-Linearity is Important:** Tree-based models (Random Forest and XGBoost) outperformed Linear Regression, indicating that non-linear relationships and complex interactions are significant in predicting flood risk within this dataset.
*   **XGBoost Superiority:** XGBoost yielded the best performance metrics among the models tested, suggesting its effectiveness in handling the complexity of the multi-domain synthetic indicators.
*   **Feature Importance Validation:** The consistent high importance of `Env_Urban_Interaction` across tree-based models validates the hypothesis that the interplay between environmental and urban factors is key to flood risk prediction.
*   **Polynomial Features in Tree Models:** While polynomial features were added, they did not show direct importance in the tree-based models. This is often because tree-based models can inherently model non-linear relationships and interactions without explicit polynomial terms, or the linear combination captured by the primary interaction term was already sufficient.
