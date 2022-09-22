import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split

train=pd.read_csv('C:\Users\ABDILLAH\Desktop\Reopened_dataset\RailsDistributionData.csv')

features_col=['First_Status','Reputation','Changed_file','Evaluation_time','Num_Comments_before_Closed','Num_commits_before_Closed','Num_lines_added','Num_lines_deleted','classes']
#define X and y
#X=train.loc[:,features_col].dropna()
X=train[features_col].dropna()
y=train.classes

#split X and y into training and testing set

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

#train a logistic regression model in the training set

logreg=LogisticRegression()
logreg.fit(X_train,y_train)

y_pred_class=logreg.predict(X_test)

#Calculate the accuracy 

print'Accuracy', metrics.accuracy_score(y_test, y_pred_class)

#Examine  the class distribution of the testing set (Using pandas series method)
y_test.value_counts()

#Calculate the percentage of ones by simply calculate the mean
y_test.mean()


#Calculate the percentage of zeros 
1-y_test.mean()

#Calculate the null accuracy (for binary classification coded as 0/1)
max(y_test.mean(),1-y_test.mean())

#Calculate the null accuracy (for multi_class classification)
y_test.value_counts().head(1)/len(y_test)

#print the first 25 true and predicted responses
print'True',y_test.values[0:30]
print 'Pred',y_pred_class[0:30]

#Confusion matrix
confusion=metrics.confusion_matrix(y_test,y_pred_class)
TP=confusion[1:1]
TN=confusion[0:0]
FP=confusion[0:1]
FN=confusion[1:0]

#![Small confusion matrix](images/confusion_matrix.png)

#Calculate the recall (Sensitivity)
#print TP / float(TP+FN)
print 'Recall', metrics.recall_score(y_test,y_pred_class)

#Calculate the precision (Specificity)
#print TN / float(TN+FP)
print'Precision', metrics.precision_score(y_test,y_pred_class)
print 'Recall', 1-metrics.recall_score(y_test,y_pred_class)

#print the first 10 predicted classes
logreg.predict(X_test)[0:14134]

#print the first 10 predicted probabilities for class 1
#logreg.predict(X_test)[0:10, 1]

#Store the predicted probabilities for class 1
y_pred_prob=logreg.predict_proba(X_test)[:]

#allow plot to appear in the netbook
#get_ipython().magic(u'matplotlib inline')

#plt.rcParams['font.size']=14


#Histogram of predicted probabilities
#plt.hist(y_pred_prob,bins=8)
#plt.xlim(0,1)
#plt.title('Histogram of predicted probability')
#plt.xlabel('Predicted probability of reopened pull request')
#plt.ylabel('Frequency')
#fig=plt.figure()
#fig.savefig('C:\Users\ABDILLAH\Desktop\datasets\Rails\Railslogreg.png',transparents=True,bbox_inches='tight')


#IMPORTANT: first argument is true value and second argument is predicted probabilities
fpr, tpr, thresholds=metrics.roc_curve(y_test,y_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('ROC Curve for reopened classifier')
plt.xlabel('False positive Rate (1-Specificity)')
plt.ylabel('True positive Rate (Sensitivity)')
plt.grid(True)

#define a function that accepts a threshold and print sensitivity and specificity
def evaluate_threshold(threshold):
    print'Sensitivity',(tpr)[threshold>threshold]
    print'Specificity',1-(fpr)[threshold>threshold]

evaluate_threshold(0.5)

#IMPORTANT: first argument is true value and second argument is predicted probabilities
print 'ROC_AUC', metrics.roc_auc_score(y_test,y_pred_prob)

#calculate cross-validate AUC
from sklearn.cross_validation import cross_val_score
print 'AUC', cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()

