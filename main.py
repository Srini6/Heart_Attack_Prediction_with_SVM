import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics

#Importing Data
heart_data = pd.read_csv('heart.csv')

#Age Plot
plt.figure()
heart_data['age'].value_counts().plot.bar(title='Age')
#plt.show()

#Data Preprocessing

X = heart_data.drop(['output'],axis=1)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)
y = heart_data.output

#Training and Test Data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.40,random_state=30)

#Training SVM
linearSVM = svm.SVC(kernel='linear',probability=True)
linearSVM.fit(X_train,y_train)

#Prediction of Test set
Z = linearSVM.predict(X_test)
predict_probabilities = linearSVM.predict_proba(X_test)

#Analysing the model
accuracy = metrics.accuracy_score(y_test, Z)
precision = metrics.precision_score(y_test,Z)
fpr,tpr,threshold = metrics.roc_curve(y_test,predict_probabilities[:,1])
roc_auc = metrics.auc(fpr,tpr)
print(f"Accuracy : {accuracy * 100} \nPrecision : {precision*100} \nAUC : {roc_auc*100}")

#Plots
#ROC Plot
plt.figure()
plt.title('ROC Plot')
plt.plot(fpr,tpr,'b')
plt.xlim([0,1])
plt.ylim([0,1])
#plt.show()

#Confusion Matrix
plt.figure()
metrics.plot_confusion_matrix(linearSVM, X_test,y_test)
plt.title('Confusion Matrix')
plt.show()