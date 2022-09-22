from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Loading dataset
train=pd.read_csv('C:/Users/ABDILLAH/Desktop/Reopened_dataset/twbsDistributionData.csv')
features_col=['First_Status','Reputation','Changed_file','Evaluation_time','Num_Comments_before_Closed','Num_commits_before_Closed','Num_lines_added','Num_lines_deleted']
X=train[features_col].dropna()
y=train.classes
test_size=0.4 #could also specify train_size=0.7 instead
random_state=0
#train_test_split convenience function
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,test_size=test_size)
clf = tree.DecisionTreeClassifier()
clf=clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
y_pred_prob=clf.predict_proba(X_test)
#Confusion matrix
print(confusion_matrix(y_test,y_pred))
print("Accuracy: {0:.2f}%".format(accuracy_score(y_test,y_pred)*100))
print("Precision: {0:.2f}%".format(metrics.precision_score(y_test,y_pred)*100))
print("Recall: {0:.2f}%".format(metrics.recall_score(y_test,y_pred)*100))
print("F-measure:{0:.2f}%".format(2*metrics.precision_score(y_test,y_pred)*metrics.recall_score(y_test,y_pred)/(metrics.precision_score(y_test,y_pred)+metrics.recall_score(y_test,y_pred))*100))
print("ROC-AUC: {0:.2f}%".format(metrics.roc_auc_score(y_test,y_pred)*100))
#print (metrics.roc_auc_score(y_test,y_pred_prob))
#Print the classification report
#print importances_feautres
from sklearn.metrics import classification_report
print (classification_report(y_test,y_pred))

importances_feautres=pd.DataFrame({'features':features_col,'importance':np.round(clf.feature_importances_,3)})
importances_feautres = importances_feautres.sort_values('importance',ascending=False).set_index('features')
#Print the classification report
print importances_feautres
plt.barh(features_col,np.round(clf.feature_importances_,3))
plt.xlabel('Relative Importance')
#plt.title('Variable Importance')
plt.show()
