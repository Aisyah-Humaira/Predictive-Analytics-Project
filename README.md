# Machine Learning Project

## Project Domain
![image](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/768bb849-7586-4158-8a52-c8f867d405b2)

Diabetes is one of the most common chronic diseases recognized worldwide and can pose serious health risks if not managed appropriately. According to data obtained around 2019, the global prevalence of diabetes was estimated at 9.3% or approximately 463 million people worldwide. Projections for the future show an increasing trend, with estimates reaching 10.2% or approximately 578 million people by 2030, and increasing again to 10.9% or approximately 700 million people by 2045. In addition, it was found that half of the individuals living with diabetes, 50.1% to be exact, are unaware that they have the disease. This shows the importance of early detection and awareness of diabetes symptoms. The impact of uncontrolled diabetes can be devastating to the body's overall health. Many of the body's organs, including the kidneys, eyes, heart, and nervous system, can suffer serious damage due to complications of diabetes that are not detected or managed properly. Therefore, it is very important to diagnose diabetes quickly and accurately to prevent serious complications and ensure a better quality of life for people with diabetes[[1]](https://www.sciencedirect.com/science/article/pii/S1110866524000045).

Many machine learning (ML) techniques are used in the medical sector to detect and predict health problems. One of the diseases for which ML techniques are used to find the most effective treatment is diabetes. Machine learning techniques are applied in almost every field of life to solve practical problems due to their ability to produce consistent, reliable, and accurate results. Thus, the use of ML in diabetes diagnosis can speed up the disease identification process, reduce the risk of human error, and ensure patients get the treatment that suits their condition[[2]](https://www.sciencedirect.com/science/article/pii/S2772442523001405).

## Business Understanding

### Problem Statements
- How to get a machine learning model to predict diabetes?
- Which model development gives the smallest error for predicting diabetes?

### Goals
- Obtain a machine learning model that can be used to predict diabetes.
- Find out the development model that gives the results with the smallest error to predict diabetes.

## Data Understanding
The dataset used in this project was obtained from Kaggle's website on [Simple Feature To Detect Diabetes](https://www.kaggle.com/datasets/simaanjali/diabetes-simple-diagnosis). This dataset consists of 88380 rows and 9 columns. 

Based on information from Kaggle, the variables in the dataset are as follows:
- ***Age***: Represents the patient's age in years. Age can be a risk factor for diabetes, as the risk of diabetes increases with age.
- ***Gender***: Indicates the gender of the patient, which can be a factor in the prediction of diabetes. Some studies suggest that women may have a different risk than men in developing diabetes.
- ***Body Mass Index (BMI)***: BMI is a measure that uses a person's height and weight to determine whether they are in the normal weight, overweight, or obese category. A high BMI is associated with a higher risk of diabetes.
- ***High Blood Pressure (High_BP)***: An indicator of whether a patient has hypertension. High blood pressure is a significant risk factor for type 2 diabetes.
- ***Fasting Blood Glucose (FBS)***: Represents the level of glucose in the blood after an overnight fast. A high fasting blood sugar level may indicate a risk of diabetes or prediabetes.
- ***HbA1c (HbA1c_level)***: A measurement of the average blood sugar level over the past 2-3 months. It is an important indicator for the diagnosis and management of diabetes.
- ***Smoking***: Indicates whether the patient smokes or not. Smoking can be an additional risk factor for type 2 diabetes.
- ***Diabetes***: An indicator that a person has diabetes.

Apart from the variable description, the following information is also obtained about the dataset- There is 1 non-numeric column with object type, namely *Gender*. This column is a *categorical feature*.
- There is 1 numeric column with float64 data type, namely *HbA1c_level*. This column is a *numerical feature*.
- There are 7 numeric columns with int64 data type, namely: *Unnamed: 0, Age, BMI, High_BP, FBS, Smoking, and Diagnosis*. These columns are *numerical features*.

## Exploratory Data Analysis (EDA)
In the next step, Exploratory Data Analysis (EDA) is conducted to understand and analyze the characteristics of the data used. EDA aims to find patterns, identify anomalies, and check the assumptions in the dataset. There are two methods used, namely univariate and multivariate methods.

1. *Univariate Analysis* involves one variate or variable. In the process of analyzing category features, the following figure displays the data obtained.
   
![grafik_gender](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/16b8144c-e1dd-44de-a677-353d791c0492)

Figure 1. Gender Chart


In Figure 1, there are 3 gender features, namely *Female, Male,* and *Other*. From the percentage data, it can be concluded that the most common gender is *Female* with a percentage of 58.10%, then gender *Male* is 41.88%, and finally *other* is only around 0.02%.

Furthermore, for the numerical features, the following figure shows the data obtained.

![image](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/991f9797-6757-4479-8a0e-d06b73870c37)

Figure 2. Histogram of Numerical Features

In Figure 2, it can be seen that the *Age* feature has the most samples at the age of 80. Then in the *BMI* feature, most samples have a body weight in the range of 30 kg. Then in the *High_BP* sample, it can be seen that most samples have high blood pressure. Then in the *FBS* feature, it can be seen that more samples have a blood glucose level of more than 150. Then for the *HbA1c_level* feature, the average blood sugar level for the last 2-3 months is mostly in the range of 6. Then in the *Smoking* feature, more samples smoke than not. Finally, in the Diagnosis feature, it can be seen that more samples were diagnosed with diabetes.

2. *Multivariate Analysis* involves two or more variables. In this process, the following figure shows the data obtained.
   
![image](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/00add511-e329-48e1-95a8-a5ca7898d393)

Figure 3. Correlation Among Features

From Figure 3, we can see the evaluation of the correlation score between features. When looking at the correlation to diagnosis, the feature that has the highest score is *FBS* or *Fasting Blood Glucose* which shows the level of glucose in the blood after fasting overnight where high fasting blood sugar levels can indicate the risk of diabetes or prediabetes. In addition, the *HbA1c_level* feature has the highest score, which is the average blood sugar level over the last 2-3 months.

## Data Preparation
In this step, data preparation is carried out which aims to transform the dataset. This transformation is done so that the data has a format or form that is appropriate and suitable for the modeling process in machine learning. Some common steps in data preparation include:

- Category Feature Encoding which is done using the *one-hot-encoding* technique. This technique is used to convert categorical variables into binary form (0 or 1) so that they can be processed by machine learning algorithms. This technique was applied to the *Gender* feature.
- Split the dataset with the *train_test_split* function so that the dataset becomes training data (*train*) and test data (*test*) with a commonly used proportion of 80:20. This technique is used to test the performance of the model on data that has not been seen before, to check whether the model is *overfitting* or generalizing well to new data. 
- Standardisation can help to make variables have a similar scale, so that distance-based or optimization *machine learning* algorithms can work more efficiently and accurately. This data transformation process changes the mean (*mean*) value to 0 and the standard deviation value to 1. The use of standardization can ensure that all features are similarly scaled, which can improve performance and enhance model interpretation. The technique used is based on the following equation. 

$$z = \frac{x - μ}{σ}$$

Notes:

x = each value in the numeric feature

μ = Average of the training samples

σ = Standard deviation of training samples

from the above calculations, the results are obtained in the following table.

Table 1. Results of Data Standardisation
|| Age | BMI | High_BP | FBS | HbA1c_level | Smoking |
| --- | --- | --- | ------- | --- | ----------- | ------- |
| **18362** | 1.700510 | 0.280882 | 3.294713 | 0.491845 | 0.602952 | -0.674932 |
| **7629** | -0.722260	 | 2.018572 | -0.303517 | 0.032739 | 0.142361 | -0.674932 |
| **24394** | 0.668589 | 2.887417 | -0.303517 | 0.395191 | 0.418716 | -0.674932 |
| **14573** | 0.354526 | 0.715304 | -0.303517 | 1.482546 | -0.686703 | 1.481630 |
| **6118** | 0.623723 | -0.008733 | -0.303517 | -0.208895 | 3.182264 | -0.674932 |


Furthermore, I checked the mean value which was changed to 0, and the standard deviation value to 1 after the standardisation process where the results were obtained in the following table.

Table 2. Descriptive Statistics After Standardisation
| | Age | BMI | High_BP | FBS | HbA1c_level | Smoking |
| --- | --- | --- | ------- | --- | ----------- | ------- |
| **count** | 70704.0000 | 70704.0000 | 70704.0000 | 70704.0000 | 70704.0000 | 70704.0000 |
| **mean** | -0.0000	| 0.0000 | -0.0000 | -0.0000 | 0.0000 | -0.0000 |
| **std** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **min** | -1.8888 | -2.4705 | -0.3035 | -1.4171 | -1.8842	 | -0.6749 |
| **25%** | -0.8120	 | -0.5880	 | -0.3035 | -0.9338	 | -0.6867 | -0.6749 |
| **50%** | 0.0405 | -0.0087 | -0.3035 | 0.0327 | 0.2345	| -0.6749 |
| **75%** | 0.8032 | 0.4257 | -0.3035 | 0.4918 | 0.6030 | 1.4816 |
| **max** | 1.7005 | 9.2589 | 3.2947 | 3.8989 | 5.9458 | 1.4816 |

## Modeling
Model development is the process of creating, training, and evaluating models to predict or classify data based on existing features. This stage will develop a machine learning model with three algorithms, namely K-Nearest Neighbor, Random Forest, and Boosting Algorithm.

### [K-Nearest Neighbor (KNN)](https://lp2m.uma.ac.id/2023/02/16/algoritma-k-nearest-neighbors-knn-pengertian-dan-penerapan/)
***K-Nearest Neighbor*** is one of the basic algorithms in machine learning used for regression and classification. In the KNN algorithm, it is assumed that similar data tends to be near or neighborhood and that data with similar characteristics will be nearby. The goal of KNN is to find the nearest neighbor of a given query point, thus we can determine the class label for that point based on the majority of the class labels of its nearest neighbors. KNN only requires two main parameters, namely the k value and the distance metric, which are relatively less in number compared to most other machine learning algorithms.

Steps of the KNN method algorithm: 
1. The KNN model for regression is initialised using *KNeighborsRegressor* with the number of nearest neighbours (*n_neighbors*) of 10.
2. The initialized KNN model is then trained using training data (*X_train*) and training labels (*y_train*). This training process aims for the model to understand the patterns and relationships between features (*X*) and labels (*y*).
3. Model performance is evaluated using the *Mean Squared Error* (*MSE*) metric.
   
### [Random Forest (RF)](https://kantinit.com/kecerdasan-buatan/random-forest-pengertian-cara-kerja-dan-contoh-penerapannya/) 
***Random Forest*** is one of the methods that has similarities with Decision Tree. It is one of the most popular algorithms due to its accuracy, simplicity, and flexibility. Its ability to be used in classification and regression, coupled with its nonlinear nature, makes it highly adaptable to different types of data and situations. To get accurate and consistent predictions, Random Forest uses the bagging method, which is a technique of combining several meta-algorithms to improve the accuracy of machine learning algorithms. This bagging method takes random samples from the dataset through a raw sampling process. After that, the samples obtained from raw sampling are reused with replacement, this process is known as bootstrapping and produces bootstrap samples. Each model is then trained independently until it produces a prediction. The final result is determined based on the majority prediction of all models. In simple terms, the predictions from each model are aggregated and then analyzed to determine the majority result. This process is called aggregation.

Steps of the Random Forest method algorithm:
1. The *Random Forest* model for regression (*RandomForestRegressor*) is initialised with the following parameters:
   - *n_estimators* = 50 (Number of decision trees in the forest)
   - *max_depth* = 16 (Maximum depth of each decision tree)
   - *random_state* = 55 (Seed for randomization to make the results reproducible)
   - *n_jobs* = -1 (Uses all available processor cores for training)
2.  The initialized *Random Forest* model is then trained using the training data and training labels.
3.  Model performance is evaluated using the *Mean Squared Error* (*MSE*) metric.
      
### [Boosting Algorithm](https://aws.amazon.com/id/what-is/boosting/) 
***Boosting Algorithm*** is a technique in machine learning that is used to reduce errors in data prediction. It improves the accuracy and performance of machine learning models by transforming multiple weak models into one strong learning model. AdaBoost (Adaptive Boosting) is one of the first developed boosting methods. In AdaBoost, each data is initially given equal weight. After each iteration or decision tree formation, the weight of each data will be adjusted automatically. More weight will be given to misclassified data to be corrected in the next iteration. This process will be repeated until the prediction error, or the difference between the true and predicted values falls below an acceptable error level.

Steps of the Boosting Algorithm method algorithm:
1. The AdaBoostRegressor model is initialized with the following parameters:
   - *learning_rate* = 0.05 (Learning rate that controls how much each weak model contributes to the final combined model)
   - *random_state* = 55 (Seed for randomization to make the results reproducible)
3. The initialized *Random Forest* model is then trained using the training data and training labels.
4.  Model performance is evaluated using the *Mean Squared Error* (MSE) metric.

## Evaluation
An evaluation metric to assess the accuracy of a regression model in predicting numerical data is the *Mean Squared Error* (MSE). MSE measures the difference between the model prediction and the actual value of the data and then squares the difference to avoid negative difference values. After that, the squared differences of each data are summed up and averaged to get a value of [MSE](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4420880).

$$MSE = \frac{1}{N} \sum_{k=1}^n \left( y_i - \hat{y} \right)^2$$

Notes:

N = number of datasets

$\hat{y}$ = actual value

$`y_i`$ = predicted value

Before calculating the MSE value in the model, it is necessary to scale the numerical features in the test data first to ensure that the numerical features in the test data have a similar scale to the train data, which has been scaled before. Next, I evaluate the three models using the MSE metric.

The evaluation results on the train data and test data are listed in the following table.

Table 3. Data Evaluation Results of the Three Models
|| train | test | 
| --- | --- | --- |
| **KNN** | 0.027474 | 0.031791 | 
| **RF** | 0.015403 | 0.026833 | 
| **Boosting** | 0.031805 | 0.030721 | 

To make it easier to see the results from Table 3, a plot of these metrics is made using a bar chart such as the results obtained in the following figure.

![image](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/c3e7fffe-dd38-4a27-bb4d-9424e34a7195)

Figure 3: Metric Evaluation Chart

From Figure 3, it can be seen that the *Random Forest* (RF) model provides the smallest error value while the model with the *K-Nearest Neighbor* (KNN) algorithm has the largest error. Therefore, the RF model can be said to be the best model for prediction because it has the smallest error.

To test this, predictions were made using several pre-trained machine-learning models on one of the data rows. This process gives an idea of how each model predicts the given data.

Table 4. Prediction Results of the Model
|| y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting | 
| --- | --- | --- | --- | --- |
| **71760** | 0 | 0.1 | 0.05 | 0.18 |

It can be seen in Table 4 that prediction with *Random Forest* (RF) gives the closest result to the true value (*y_true*).

## Conclusion
From the Applied Machine Learning project that has been done regarding the prediction of diabetes, it can be concluded:

1. To develop a *machine learning* model for predicting diabetes, some algorithms that can be used include *K-Nearest Neighbor* (KNN), *Random Forest, and Boosting Algorithm*.
2. Of the three algorithms, the evaluation results show that the *Random Forest* algorithm produces the smallest error compared to KNN and *Boosting Algorithm*. In addition, *Random Forest* also produces predictions that are closest to the actual value.
   
## References
[1] B. A. N.G., "En-RfRsK: An ensemble machine learning technique for prognostication of diabetes mellitus," Egyptian Informatics Journal, vol. 25, 2024. (https://doi.org/10.1016/j.eij.2024.100441)

[2] S. S. Bhat, M. Banu, G. A. Ansari and V. Selvam, "A risk assessment and prediction framework for diabetes mellitus using machine learning algorithms," Healthcare Analytics, vol. 4, 2023. (https://doi.org/10.1016/j.health.2023.100273)
