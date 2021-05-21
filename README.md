# Heart_Attack_Prediction_with_SVM

This project is made from the dataset publiched in Kaggle under the link https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

**Understanding Dataset**

Dataset contains 14 columns. Their details are as follows.

About this dataset Age : Age of the patient

Sex : Sex of the patient

exang: exercise induced angina (1 = yes; 0 = no)

ca: number of major vessels (0-3)

cp : Chest Pain type chest pain type

Value 1: typical angina Value 2: atypical angina Value 3: non-anginal pain Value 4: asymptomatic trtbps : resting blood pressure (in mm Hg)

chol : cholestoral in mg/dl fetched via BMI sensor

fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

rest_ecg : resting electrocardiographic results

Value 0: normal Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

thalach : maximum heart rate achieved

target : 0= less chance of heart attack 1= more chance of heart attack

Source : Kaggle

**Data Preprocessing and Data Split up**
Standardizing

Data features measured in different scales will carry a bias to the model. Standardizing the features allows rescaling of the elements in the column to be centered around 0. This brings a common scale to all the columns of the dataset. StandardScaler from Scikit-learn preprocessing is applied on each feature of the dataset. Reference : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

Training and Test set

Dataset is split into training and test set with random data points. test_size represents the size of the test set. In this case, 40 percent of data is classified as test set. random_state shuffles the dataset before spliting. Integer value of random_state is helpful to repreduce the same training and test set on multiple execution. Reference : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

**Training SVM**
SVMs are based on finding a hyperplane that applies best for classification of the dataset. SVMs are advantageous in training with small dataset and gives an accurate prediction. But on a dataset with overlapping features, SVMs are less effective.

**Analysing the trained model**
To analyse the model, accuracy, precision and ROC curve is generated using Sklearn metrics.

**Conclusion**
With SVM Linear kernal, a model with accuracy of 86% and precision with 89% is modelled. Further, ML performance improvement techniques can be implemented to improve the accuracy rate of the model.
