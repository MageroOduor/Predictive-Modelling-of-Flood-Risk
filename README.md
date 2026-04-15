# Predictive-Modelling-of-Flood-Risk

## **Problem Statement**

Rapidly growing cities worldwide face increasing flood risks due to complex interactions between environmental factors (e.g., climate change, monsoon intensity) and urban development (e.g., urbanization, inadequate drainage systems). However, real-world flood data is often scarce or inconsistent, making accurate risk prediction a significant challenge. This project addresses the critical need for robust flood risk prediction by performing a case study using a synthetic, multi-domain environmental indicator dataset. The goal is to develop and evaluate machine learning models that can effectively identify high-risk areas and influential factors, providing a framework that can inform disaster preparedness, urban planning, and resource allocation strategies, ultimately aiming to mitigate the socio-economic impacts of flooding, even when real-world data limitations exist.

## **General Objective**

To develop, evaluate, and interpret an accurate machine learning model capable of predicting flood probability with an RMSE below 0.015 and an R-squared above 0.90 using a synthetic multi-domain environmental dataset, thereby providing actionable insights for urban planning and disaster preparedness within the project timeline.

## **Specific Objectives**

1. Conduct comprehensive Exploratory Data Analysis (EDA) to understand feature distributions, identify correlations, and detect potential anomalies in the synthetic dataset.
2. Create meaningful composite features (e.g., environmental and urban indices, interaction terms, polynomial features) and preprocess the data (e.g., scaling) to optimize it for machine learning models.
3. Build and evaluate multiple machine learning models (e.g., Linear Regression, Random Forest, XGBoost) to predict flood probability, comparing their performance using metrics like RMSE and R-squared.
4. Interpret the best-performing model's outputs and feature importances to identify the most influential predictors of flood risk and discuss the implications of the findings.

## **Significance of the Project**

This project demonstrates the practical value of machine learning in addressing real-world flood risks by: 
1. Providing a reliable, data-driven approach to predicting flood risk in rapidly growing urban areas;
2. Supporting better urban planning by identifying high-risk areas and guiding infrastructure decisions.
3. Enhancing disaster preparedness through early risk detection and improved response planning.
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
