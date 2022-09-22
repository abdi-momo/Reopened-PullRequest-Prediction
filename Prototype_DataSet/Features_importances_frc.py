import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
#import some data to play with
train=pd.read_csv('C:/Users/ABDILLAH/Desktop/Reopened_dataset/twbsDistributionData.csv')
features_col=['First_Status','Reputation','Changed_file','Evaluation_time','Num_Comments_before_Closed','Num_commits_before_Closed','Num_lines_added','Num_lines_deleted']
#define X and y
X=train[features_col].dropna()
y=train.classes
#class_names = train.target_names
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
clf = RandomForestClassifier(max_depth=1, random_state=0)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#Confusion matrix
print(confusion_matrix(y_test,y_pred))
print("Accuracy: {0:.2f}%".format(metrics.accuracy_score(y_test,y_pred)*100))
print("Precision: {0:.2f}%".format(metrics.precision_score(y_test,y_pred)*100))
print("Recall: {0:.2f}%".format(metrics.recall_score(y_test,y_pred)*100))
print("F-measure:{0:.2f}%".format(2*metrics.precision_score(y_test,y_pred)*metrics.recall_score(y_test,y_pred)/(metrics.precision_score(y_test,y_pred)+metrics.recall_score(y_test,y_pred))*100))
print("ROC-AUC: {0:.2f}%".format(1-metrics.roc_auc_score(y_test,y_pred)*100))

#Print the classification report
from sklearn.metrics import classification_report
print (classification_report(y_test,y_pred))

importances_feautres=pd.DataFrame({'features':features_col,'importance':np.round(clf.feature_importances_,3)})
importances_feautres = importances_feautres.sort_values('importance',ascending=False).set_index('features')
#Print the classification report
print importances_feautres
plt.barh(features_col,np.round(clf.feature_importances_,3))
plt.show()
