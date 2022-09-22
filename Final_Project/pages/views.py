from django.contrib.auth.decorators import permission_required
from pages.models import Pull_Requests, rails_Project,cocos2d_Project,symfony_Project,caskroom_Project, \
	zendframework_Project, angular_Project,bootstrap_Project
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Q
from django.shortcuts import render, render_to_response,get_object_or_404, _get_queryset, redirect
from django.contrib import messages
from django.contrib.auth import (authenticate,get_user_model, login, logout)
from django.contrib.auth.models import User
from django import forms
from io import StringIO
import csv
from django.urls import reverse


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import LeaveOneOut
import json
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pages.forms import UserForm, UserProfileForm
# from django.contrib.auth import authenticate, login
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
from django.views.generic import TemplateView


def model_picke():
	train = pd.read_csv('F:/django-project/Final_Project/pages/dataset/zendframework.csv')
	features_col = ['First_Status', 'Reputation', 'Changed_file', 'Evaluation_time', 'Num_Comments_before_Closed',
					'Num_commits_before_Closed', 'Num_lines_added', 'Num_lines_deleted']
	X = train[features_col].dropna()
	y = train.classes
	test_size = 0.2  # could also specify train_size=0.7 instead
	train_size = 0.7
	random_state = 0
	# train_test_split convenience function
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred_prob = clf.predict_proba(X_test)
	# Confusion matrix
	print(confusion_matrix(y_test, y_pred))
	print("Accuracy: {0:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
	print("Precision: {0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100))
	print("Recall: {0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100))
	print("F-measure:{0:.2f}%".format(
		2 * metrics.precision_score(y_test, y_pred) * metrics.recall_score(y_test, y_pred) / (
					metrics.precision_score(y_test, y_pred) + metrics.recall_score(y_test, y_pred)) * 100))
	print("ROC-AUC: {0:.2f}%".format(metrics.roc_auc_score(y_test, y_pred) * 100))

	from sklearn.metrics import classification_report
	print(classification_report(y_test, y_pred))

	importances_feautres = pd.DataFrame({'features': features_col, 'importance': np.round(clf.feature_importances_, 3)})
	importances_feautres = importances_feautres.sort_values('importance', ascending=False).set_index('features')


# Uploading the prediction dataset from the newly downloaded csv file
@permission_required('admin.can_add_log_entry')
def upload_file(request, *args, **kwargs):
	template='upload_file.html'
	if request.method == 'GET':
		return render(request, template)
	CSV_file=request.FILES['csv_file']

	if not CSV_file.name.endswith('.csv'):
		messages.error(request, 'This is not a CSV file')
		return HttpResponseRedirect(reverse('add_pull_requests'))
	# data_set=CSV_file.read().decode('UTF-8')

	train=pd.read_csv(CSV_file) #This is a test to know if it will work

	features_col = ['Comments', 'LC_added', 'LC_deleted', 'Commits', 'Changed_files', 'Evaluation_time','First_status','Reputation'] # This also test
	class_label=['Label']
	X = train[features_col] # This also test
	y=train[class_label]

	# Replace the Pandas DataFrames Values
	for items in X['First_status']:
		# print(items)
		if items=="Accepted":
		# if "Accepted" in X['First_status']:
			X['First_status'].replace('Accepted', 1, inplace=True)
		elif items=="Rejected":
		# elif "Rejected" in X['First_status']:
			X['First_status'].replace('Rejected', 0, inplace=True)
		else:
			pass
		# X['First_status'].replace(['Accepted', 'Rejected'],[1,0], inplace=True)
	recommendSet=[]
	actualSet=[]
	for rows in y['Label']:
		# print(rows)
		if rows=="Non-Reopened":
			print(rows)
			y['Label'].replace('Non-Reopened', 0, inplace=True)
			recommendSet.append(rows)
		elif rows=="Reopened":
			y['Label'].replace('Reopened', 1, inplace=True)
			actualSet.append(rows)
		else:
			pass
	new_data=[X,y]
	# print(X)
	# print(y)
	random_state = 0
	# test_size=request.GET.get('test_size')
	# for train_index, test_index in loo.split(X):
	# 	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	# 	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.2)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	try:
		Accuracy="{0:.2f}%".format(accuracy_score(y_test, y_pred) * 100)
		Precision="{0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100)
		Recall="{0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100)
		F1_meseaure="{0:.2f}%".format(2*metrics.precision_score(y_test,y_pred)*metrics.recall_score(y_test,y_pred)/(metrics.precision_score(y_test,y_pred)+metrics.recall_score(y_test,y_pred))*100)
		# F1_meseaure=("{0:.2f}%".format(2*Precision*Recall)/(Precision+Recall))*100
	except ZeroDivisionError:
		print("Error: dividing by zero")
		F1_meseaure='nan%'
    #
	# print(accuracy_score(y_test,y_pred)*100)
	print("Accuracy: {0:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
	print("Precision: {0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100))
	print("Recall: {0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100))
	print("F1-measure: ", F1_meseaure)
	importances_feautres = pd.DataFrame({'features': features_col, 'importance': np.round(clf.feature_importances_, 3)})
	importances_feautres = importances_feautres.sort_values('importance', ascending=False).set_index('features')
	# print(importances_feautres.shape)
	importances_feautres = [ls[0] for ls in importances_feautres.values.tolist()]
	classification_report={'accuracy':Accuracy, 'pricision':Precision, 'recall':Recall, 'f1_score':F1_meseaure}
	importance_features={'importances_feautre':importances_feautres}

	plt.barh(features_col, np.round(clf.feature_importances_, 3))
	plt.xlabel('Relative Importance')
	plt.title('Variable Importance')
	plt.show()
	data={
		'new_data':new_data,
 		'classification_report':classification_report,
		'importance_feature':importance_features,
		'features':features_col,
	 }

	return render(request,template, data)


def resultOfPredicting(request):
	pull_requestsList = rails_Project.objects.all()
	project = request.GET.get('project')
	prId = request.GET.get('id')
	newPredictionData = {}

	if pull_requestsList.filter(pr_project=project):
		# indexId=pull_requestsList.filter(pr_project=project).valuest('pr_project')
		q = pull_requestsList.filter(pr_id=prId).values('pk', 'pr_project', 'pr_id', 'nb_comments','nb_added_lines_code', 'nb_deleted_lines_code', 'nb_commits',
														'nb_changed_fies', 'time_evaluation', 'Closed_status','reputation', 'Label')

	# queryList = list(q)

	dict = q[0]
	print(dict)
	print('----------------------')
	print(dict['pk'])

	prIndex = dict['pk']

	newPredictionData = {'index': dict['pk'], 'project': dict['pr_project'], 'prId': dict['pr_id'],
						 'comments': dict['nb_comments'], 'lc_added': dict['nb_added_lines_code'],
						 'lc_deleted': dict['nb_deleted_lines_code'], 'commits': dict['nb_commits'],
						 'changed_files': dict['nb_changed_fies'],
						 'evaluation_time': dict['time_evaluation'], 'firstStatus': dict['Closed_status'],
						 'reputation': dict['reputation'],
						 'classLabel': dict['Label']}
	data_list = list(newPredictionData.values())
	print('-----------------')
	print(data_list)
	print('-----------------')
	for i in range(len(data_list)):
		if data_list[i] == "Reopened":
			data_list[i] = 1
		elif data_list[i] == "Non-Reopened":
			data_list[i] = 0
		elif data_list[i] == "Rejected":
			data_list[i] = 0
		elif data_list[i] == "Accepted":
			data_list[i] = 1
		else:
			pass
	# y = data[-1]

	for j in range(len(data_list)):
		newData = data_list[0:j]
	# print('-----------------')
	# print('New data', newData)
	train = pd.read_csv('F:/django-project/Final_Project/pages/dataset/RailsDistributionData.csv')
	features_col = ['First_Status', 'Reputation', 'Changed_file', 'Evaluation_time', 'Nb_Comments',
					'Nb_Commits', 'LC_added', 'LC_deleted']
	X = train[features_col].dropna()
	y = train.classes
	test_size = 0.2  # could also specify train_size=0.7 instead
	# train_size = 0.8
	random_state = 0
	# train_test_split convenience function
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred_prob = clf.predict_proba(X_test)
	# Confusion matrix

	try:
		Accuracy = "{0:.2f}%".format(accuracy_score(y_test, y_pred) * 100)
		Precision = "{0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100)
		Recall = "{0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100)
		F1_meseaure = "{0:.2f}%".format(
			2 * metrics.precision_score(y_test, y_pred) * metrics.recall_score(y_test, y_pred) / (
					metrics.precision_score(y_test, y_pred) + metrics.recall_score(y_test, y_pred)) * 100)
	# F1_meseaure=("{0:.2f}%".format(2*Precision*Recall)/(Precision+Recall))*100
	except ZeroDivisionError:
		print("Error: dividing by zero")
		F1_meseaure = 'nan%'

	importances_feautres = pd.DataFrame({'features': features_col, 'importance': np.round(clf.feature_importances_, 3)})
	importances_feautres = importances_feautres.sort_values('importance', ascending=False).set_index('features')
	importances_feautres = [ls[0] for ls in importances_feautres.values.tolist()]

	with open('F:/django-project/Final_Project/pages/dataset/rails_pickle', 'wb') as rails:
		pickle.dump(clf, rails)

	with open('F:/django-project/Final_Project/pages/dataset/rails_pickle', 'rb') as rails:
		# prIndex = queryList[i]['pk']
		mp = pickle.load(rails)

	score = mp.score(X_test, y_test)
	print("Test score: {0:.2f} %".format(100 * score))
	Ypredict = mp.predict(X_test)
	print(Ypredict[1])
	mp.predict(X[0:prIndex+5])
	print(y[1])


	classification_report = {'accuracy': Accuracy, 'pricision': Precision, 'recall': Recall, 'f1_score': F1_meseaure}

	importance_features = {'importances_feautre': importances_feautres}

	data = {
		'new_data': newPredictionData,
		'classification_report': classification_report,
		'importance_feature': importance_features,
		'features': features_col,
		# 'projects':projects
	}
	# {"pr_list":pull_requestsList, "new_data":newData}
	return render(request, 'resultOfPredicting.html', data)


#Affichage des données existantes dans la base de données et création des pagginations.
def index(request):
    limit=12
    pull_requestsList = rails_Project.objects.all()
    paginator = Paginator(pull_requestsList, limit)
    page=request.GET.get('page')
    try:
        pullrequests=paginator.page(page)
    except PageNotAnInteger:
        pullrequests=paginator.page(1)
    except EmptyPage:
        pullrequests=paginator.page(paginator.num_pages)
    return render(request, 'index.html', {'pullrequests': pullrequests, 'pull_requestsList':pull_requestsList})


def RailsIndex(request):
    limit=12
    pull_requestsList = rails_Project.objects.all()
    paginator = Paginator(pull_requestsList, limit)
    page=request.GET.get('page')
    try:
        pullrequests=paginator.page(page)
    except PageNotAnInteger:
        pullrequests=paginator.page(1)
    except EmptyPage:
        pullrequests=paginator.page(paginator.num_pages)
    return render(request, 'RailsIndex.html', {'pullrequests': pullrequests, 'pull_requestsList':pull_requestsList})



def home(request):
	return render(request, 'home.html', {})

from django.contrib.auth.decorators import permission_required



@permission_required('admin.can_add_log_entry')

def pull_request_upload(request):
	template="add_pull_requests.html"
	prompt={
		'order':'Order of the CSV file should be project name, Pr id, reputation, changed file, evaluation time, comments, commits, lc added, lc deleted, class label'
	}
	if request.method == 'GET':
		return render(request, template, prompt)

	CSV_file=request.FILES['file']
	if not CSV_file.name.endswith('.csv'):
		messages.error(request, 'This is not a CSV file')
		# return HttpResponseRedirect(reverse('add_pull_requests'))
	data_set=CSV_file.read().decode('UTF-8')
	io_string=StringIO(data_set)
	next(io_string)
	for column in csv.reader(io_string, delimiter=','):
		_, created=Pull_Requests.objects.update_or_create(
		pr_project=column[0],pr_id = column[1], nb_comments = column[6], nb_added_lines_code = column[8],nb_deleted_lines_code = column[9],  nb_commits = column[7], nb_changed_fies = column[4],
			time_evaluation=column[5], Closed_status= column[2], reputation= column[3], Label = column[10]
		)
	context={}
	return render(request,template,context)

def railsProject(request):
	template="railsProject.html"
	prompt={
		'order':'Order of the CSV file should be project name, Pr id, nb_comments, reputation, changed file, evaluation time, comments, commits, lc added, lc deleted, class label'
	}
	if request.method == 'GET':
		return render(request, template, prompt)

	CSV_file=request.FILES['file']
	if not CSV_file.name.endswith('.csv'):
		messages.error(request, 'This is not a CSV file')
		# return HttpResponseRedirect(reverse('add_pull_requests'))
	data_set=CSV_file.read().decode('UTF-8')
	io_string=StringIO(data_set)
	next(io_string)
	for column in csv.reader(io_string, delimiter=','):
		_, created=rails_Project.objects.update_or_create(
			pr_project=column[0],pr_id = column[1], Closed_status= column[2],reputation= column[3],nb_changed_fies = column[4],time_evaluation=column[5],
			nb_comments = column[6],nb_commits = column[7],nb_added_lines_code = column[8], nb_deleted_lines_code = column[9], Label = column[10]
		)
	context={}
	return render(request,template,context)

def cocos2dProject_Upload(request):
	template="cocos2d_Upload.html"
	prompt={
		'order':'Order of the CSV file should be project name, Pr id, nb_comments, reputation, changed file, evaluation time, comments, commits, lc added, lc deleted, class label'
	}
	if request.method == 'GET':
		return render(request, template, prompt)

	CSV_file=request.FILES['file']
	if not CSV_file.name.endswith('.csv'):
		messages.error(request, 'This is not a CSV file')
		# return HttpResponseRedirect(reverse('add_pull_requests'))
	data_set=CSV_file.read().decode('UTF-8')
	io_string=StringIO(data_set)
	next(io_string)
	for column in csv.reader(io_string, delimiter=','):
		_, created=cocos2d_Project.objects.update_or_create(
			pr_project=column[0],pr_id = column[1], Closed_status= column[2],reputation= column[3],nb_changed_fies = column[4],time_evaluation=column[5],
			nb_comments = column[6],nb_commits = column[7],nb_added_lines_code = column[8], nb_deleted_lines_code = column[9], Label = column[10]
		)
	context={}
	return render(request,template,context)


def SymfonyProject_Upload(request):
	template="symfony_Upload.html"
	prompt={
		'order':'Order of the CSV file should be project name, Pr id, nb_comments, reputation, changed file, evaluation time, comments, commits, lc added, lc deleted, class label'
	}
	if request.method == 'GET':
		return render(request, template, prompt)

	CSV_file=request.FILES['file']
	if not CSV_file.name.endswith('.csv'):
		messages.error(request, 'This is not a CSV file')
		# return HttpResponseRedirect(reverse('add_pull_requests'))
	data_set=CSV_file.read().decode('UTF-8')
	io_string=StringIO(data_set)
	next(io_string)
	for column in csv.reader(io_string, delimiter=','):
		_, created=symfony_Project.objects.update_or_create(
			pr_project=column[0],pr_id = column[1], Closed_status= column[2],reputation= column[3],nb_changed_fies = column[4],time_evaluation=column[5],
			nb_comments = column[6],nb_commits = column[7],nb_added_lines_code = column[8], nb_deleted_lines_code = column[9], Label = column[10]
		)
	context={}
	return render(request,template,context)


def CaskroomProject_Upload(request):
	template="caskroom_Upload.html"
	prompt={
		'order':'Order of the CSV file should be project name, Pr id, nb_comments, reputation, changed file, evaluation time, comments, commits, lc added, lc deleted, class label'
	}
	if request.method == 'GET':
		return render(request, template, prompt)

	CSV_file=request.FILES['file']
	if not CSV_file.name.endswith('.csv'):
		messages.error(request, 'This is not a CSV file')
		# return HttpResponseRedirect(reverse('add_pull_requests'))
	data_set=CSV_file.read().decode('UTF-8')
	io_string=StringIO(data_set)
	next(io_string)
	for column in csv.reader(io_string, delimiter=','):
		_, created=caskroom_Project.objects.update_or_create(
			pr_project=column[0],pr_id = column[1], Closed_status= column[2],reputation= column[3],nb_changed_fies = column[4],time_evaluation=column[5],
			nb_comments = column[6],nb_commits = column[7],nb_added_lines_code = column[8], nb_deleted_lines_code = column[9], Label = column[10]
		)
	context={}
	return render(request,template,context)


def Zendframework_Upload(request):
	template="Zendframework_Upload.html"
	prompt={
		'order':'Order of the CSV file should be project name, Pr id, nb_comments, reputation, changed file, evaluation time, comments, commits, lc added, lc deleted, class label'
	}
	if request.method == 'GET':
		return render(request, template, prompt)

	CSV_file=request.FILES['file']
	if not CSV_file.name.endswith('.csv'):
		messages.error(request, 'This is not a CSV file')
		# return HttpResponseRedirect(reverse('add_pull_requests'))
	data_set=CSV_file.read().decode('UTF-8')
	io_string=StringIO(data_set)
	next(io_string)
	for column in csv.reader(io_string, delimiter=','):
		_, created=zendframework_Project.objects.update_or_create(
			pr_project=column[0],pr_id = column[1], Closed_status= column[2],reputation= column[3],nb_changed_fies = column[4],time_evaluation=column[5],
			nb_comments = column[6],nb_commits = column[7],nb_added_lines_code = column[8], nb_deleted_lines_code = column[9], Label = column[10]
		)
	context={}
	return render(request,template,context)

def Angular_Upload(request):
	template="angular_Upload.html"
	prompt={
		'order':'Order of the CSV file should be project name, Pr id, nb_comments, reputation, changed file, evaluation time, comments, commits, lc added, lc deleted, class label'
	}
	if request.method == 'GET':
		return render(request, template, prompt)

	CSV_file=request.FILES['file']
	if not CSV_file.name.endswith('.csv'):
		messages.error(request, 'This is not a CSV file')
		# return HttpResponseRedirect(reverse('add_pull_requests'))
	data_set=CSV_file.read().decode('UTF-8')
	io_string=StringIO(data_set)
	next(io_string)
	for column in csv.reader(io_string, delimiter=','):
		_, created=angular_Project.objects.update_or_create(
			pr_project=column[0],pr_id = column[1], Closed_status= column[2],reputation= column[3],nb_changed_fies = column[4],time_evaluation=column[5],
			nb_comments = column[6],nb_commits = column[7],nb_added_lines_code = column[8], nb_deleted_lines_code = column[9], Label = column[10]
		)
	context={}
	return render(request,template,context)

def Bootstrap_Upload(request):
	template="bootstrap_Upload.html"
	prompt={
		'order':'Order of the CSV file should be project name, Pr id, nb_comments, reputation, changed file, evaluation time, comments, commits, lc added, lc deleted, class label'
	}
	if request.method == 'GET':
		return render(request, template, prompt)

	CSV_file=request.FILES['file']
	if not CSV_file.name.endswith('.csv'):
		messages.error(request, 'This is not a CSV file')
		# return HttpResponseRedirect(reverse('add_pull_requests'))
	data_set=CSV_file.read().decode('UTF-8')
	io_string=StringIO(data_set)
	next(io_string)
	for column in csv.reader(io_string, delimiter=','):
		_, created=bootstrap_Project.objects.update_or_create(
			pr_project=column[0],pr_id = column[1], Closed_status= column[2],reputation= column[3],nb_changed_fies = column[4],time_evaluation=column[5],
			nb_comments = column[6],nb_commits = column[7],nb_added_lines_code = column[8], nb_deleted_lines_code = column[9], Label = column[10]
		)
	context={}
	return render(request,template,context)




# def login_view(request):
# 	title="Login"
# 	form=UserForm(request.POST or None)
# 	if form.is_valid():
# 		username=form.cleaned_data.get("username")
# 		password = form.cleaned_data.get("password")
# 		user = authenticate(username=username, password=password)
# 		login(request, user)
# 		return redirect('/index')
# 		# print(request.user.is_authenticated())
# 	return render(request, 'form.html', {"form": form, "title": title})


def register_view(request):
	title = "Register"
	form =UserProfileForm(request.POST or None)
	if form.is_valid():
		user=form.save(commit=False)
		username = request.POST.get('username')
		password=form.cleaned_data.get('password')
		user.set_password(password)
		if not (User.objects.filter(username=username).exists()):
			new_user = authenticate(username=user.username, password=password)
			user.save()
			login(request, new_user)
			return redirect('/login')
		else:
			raise forms.ValidationError('Looks like a username with that email or password already exists')

	context={"form": form, "title": title}
	return render(request, 'form.html', context)

def logout_view(request):
	logout(request)
	# return render(request,'form.html', {})
	return redirect('/login')


def login_view(request):
	title = "Login"
	form = UserForm(request.POST or None)
	if form.is_valid():
	# if request.method=='POST':
		username=request.POST.get('username')
		password=request.POST.get('password')
		user=authenticate(username=username, password=password)

		if user:
			if user.is_active:
				login(request, user)
				# return HttpResponseRedirect(reverse('/index'))
				return redirect('/index')
			else:
				return HttpResponse("You account is disabled")
		else:
			print("Invalid login details: {0}, {1}", format(username,password))
			return HttpResponse("Invalid login details supplied. ")
	else:
		return render(request, 'form.html', {"form": form, "title": title})



# def get_projects(request):
# 	pull_requestsList = Pull_Requests.objects.all()
# 	project = request.GET.get('project')
# 	prId = request.GET.get('id')
# 	project1 = 'rails'
# 	project2 = 'cocos2d-x'
# 	project3 = 'symfony'
# 	project4 = 'homebrew-cask'
# 	project5 = 'zendframework'
# 	project6 = 'angular.js'
# 	project7 = 'bootstrap'
#
# 	if project:
# 		pull_requestsList = pull_requestsList.filter(
# 			Q(pr_project=project) |
# 			Q(pr_id=project) |
# 			Q(nb_comments=project) |
# 			Q(nb_added_lines_code=project) |
# 			Q(nb_deleted_lines_code=project) |
# 			Q(nb_commits=project) |
# 			Q(nb_changed_fies=project) |
# 			Q(time_evaluation=project) |
# 			Q(Closed_status=project) |
# 			Q(reputation=project)).distinct()
#
# 	    # print(pull_requestsList)
# 	if pull_requestsList.filter(pr_project=project, pr_id=prId):
# 		indexId = pull_requestsList.filter(pr_id=prId).values_list('pk')
# 		q = pull_requestsList.filter(pr_id=prId).values('pk', 'pr_project', 'pr_id', 'nb_comments',
# 														'nb_added_lines_code', 'nb_deleted_lines_code',
# 														'nb_commits', 'nb_changed_fies', 'time_evaluation',
# 														'Closed_status', 'reputation', 'Label')
# 	queryList = list(q)
#
# 	for i in range(len(queryList)):
#
# 		if queryList[i]['pr_project'] == project1:
# 			print(queryList[i]['pr_project'])
# 		elif queryList[i]['pr_project']==project2:
# 			print(queryList[i]['pr_project'])
# 		elif queryList[i]['pr_project'] == project3:
# 			print(queryList[i]['pr_project'])
# 		elif queryList[i]['pr_project']==project4:
# 			print(queryList[i]['pr_project'])
# 		elif queryList[i]['pr_project']==project5:
# 			print(queryList[i]['pr_project'])
# 		elif queryList[i]['pr_project']==project6:
# 			print(queryList[i]['pr_project'])
# 		elif queryList[i]['pr_project']==project7:
# 			print(queryList[i]['pr_project'])
# 		else:
# 			pass
# 		context={'pr_filter_list':queryList}
# 	return render(request, 'pr_filter_list.html', {'pr_filter_list':pull_requestsList})


def railsPrediction(request):

	train = pd.read_csv('F:/django-project/Final_Project/pages/dataset/RailsDistributionData.csv')
	features_col = ['First_Status', 'Reputation', 'Changed_file', 'Evaluation_time', 'Nb_Comments',
					'Nb_Commits', 'LC_added', 'LC_deleted']
	X = train[features_col].dropna()
	y = train.classes
	test_size = 0.2  # could also specify train_size=0.7 instead
	# train_size = 0.8
	random_state = 0
	# train_test_split convenience function
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred_prob = clf.predict_proba(X_test)
	# Confusion matrix

	try:
		Accuracy = "{0:.2f}%".format(accuracy_score(y_test, y_pred) * 100)
		Precision = "{0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100)
		Recall = "{0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100)
		F1_meseaure = "{0:.2f}%".format(
			2 * metrics.precision_score(y_test, y_pred) * metrics.recall_score(y_test, y_pred) / (
					metrics.precision_score(y_test, y_pred) + metrics.recall_score(y_test, y_pred)) * 100)
	except ZeroDivisionError:
		print("Error: dividing by zero")
		F1_meseaure = 'nan%'

	importances_feautres = pd.DataFrame({'features': features_col, 'importance': np.round(clf.feature_importances_, 3)})
	importances_feautres = importances_feautres.sort_values('importance', ascending=False).set_index('features')
	importances_feautres = [ls[0] for ls in importances_feautres.values.tolist()]

	with open('F:/django-project/Final_Project/pages/dataset/rails_pickle', 'wb') as rails:
		pickle.dump(clf, rails)

	with open('F:/django-project/Final_Project/pages/dataset/rails_pickle', 'rb') as rails:
		mp = pickle.load(rails)

	score = mp.score(X_test, y_test)
	print("Test score: {0:.2f} %".format(100 * score))
	Ypredict = mp.predict(X_test)
	print(Ypredict[1])

	classification_report = {'accuracy': Accuracy, 'pricision': Precision, 'recall': Recall, 'f1_score': F1_meseaure}

	importance_features = {'importances_feautre': importances_feautres}
	print('-------Importance features----------')
	for k in importance_features:
		print(importance_features[k])
	print('end')
	plt.barh(features_col, np.round(clf.feature_importances_, 3))
	labels=plt.xlabel('Relative Importance')
	plt.title('Feature Importance for the project rails')
	plt.show()

	data = {
		'classification_report': classification_report,
		'importance_feature': importance_features,
		'features': features_col,
		'labels':labels
	}
	return render(request, 'railsPredicting.html', data)



def symfonyPrediction(request):
	train = pd.read_csv('F:/django-project/Final_Project/pages/dataset/symfonyDistributionData.csv')
	features_col = ['First_Status', 'Reputation', 'Changed_file', 'Evaluation_time', 'Nb_Comments',
					'Nb_Commits', 'LC_added', 'LC_deleted']
	X = train[features_col].dropna()
	y = train.classes
	test_size = 0.2  # could also specify train_size=0.7 instead
	# train_size = 0.8
	random_state = 0
	# train_test_split convenience function
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred_prob = clf.predict_proba(X_test)
	# Confusion matrix

	try:
		Accuracy = "{0:.2f}%".format(accuracy_score(y_test, y_pred) * 100)
		Precision = "{0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100)
		Recall = "{0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100)
		F1_meseaure = "{0:.2f}%".format(
			2 * metrics.precision_score(y_test, y_pred) * metrics.recall_score(y_test, y_pred) / (
					metrics.precision_score(y_test, y_pred) + metrics.recall_score(y_test, y_pred)) * 100)
	except ZeroDivisionError:
		print("Error: dividing by zero")
		F1_meseaure = 'nan%'

	importances_feautres = pd.DataFrame({'features': features_col, 'importance': np.round(clf.feature_importances_, 3)})
	importances_feautres = importances_feautres.sort_values('importance', ascending=False).set_index('features')
	importances_feautres = [ls[0] for ls in importances_feautres.values.tolist()]

	with open('F:/django-project/Final_Project/pages/dataset/symfony_pickle', 'wb') as symfony:
		pickle.dump(clf, symfony)

	with open('F:/django-project/Final_Project/pages/dataset/symfony_pickle', 'rb') as symfony:
		mp = pickle.load(symfony)

	score = mp.score(X_test, y_test)
	print("Test score: {0:.2f} %".format(100 * score))
	Ypredict = mp.predict(X_test)
	print(Ypredict[1])

	classification_report = {'accuracy': Accuracy, 'pricision': Precision, 'recall': Recall, 'f1_score': F1_meseaure}
	importance_features = {'importances_feautre': importances_feautres}
	print('-------Importance features----------')
	for k in importance_features:
		Features_list=importance_features[k]
		# for i in Features_list:
			# print(i)
		# print(Features_list)
	print('end')
	plt.barh(features_col, np.round(clf.feature_importances_, 3))
	labels = plt.xlabel('Relative Importance')
	plt.title('Features Importance for the project symfony')
	plt.subplots_adjust(left=0.25)
	plt.savefig('F:\django-project\Final_Project\static\media\symfony_feature.png')
	# plt.show()

	data = {
		'classification_report': classification_report,
		'importance_feature': Features_list,
		'features': features_col,
	}
	return render(request, 'symfonyPredicting.html', data)



def cocos2dPrediction(request):

	train = pd.read_csv('F:/django-project/Final_Project/pages/dataset/cocos2dDistributionData.csv')
	features_col = ['First_Status', 'Reputation', 'Changed_file', 'Evaluation_time', 'Nb_Comments',
					'Nb_Commits', 'LC_added', 'LC_deleted']
	X = train[features_col].dropna()
	y = train.classes
	test_size = 0.2  # could also specify train_size=0.7 instead
	# train_size = 0.8
	random_state = 0
	# train_test_split convenience function
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred_prob = clf.predict_proba(X_test)
	# Confusion matrix

	try:
		Accuracy = "{0:.2f}%".format(accuracy_score(y_test, y_pred) * 100)
		Precision = "{0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100)
		Recall = "{0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100)
		F1_meseaure = "{0:.2f}%".format(
			2 * metrics.precision_score(y_test, y_pred) * metrics.recall_score(y_test, y_pred) / (
					metrics.precision_score(y_test, y_pred) + metrics.recall_score(y_test, y_pred)) * 100)
	except ZeroDivisionError:
		print("Error: dividing by zero")
		F1_meseaure = 'nan%'

	importances_feautres = pd.DataFrame({'features': features_col, 'importance': np.round(clf.feature_importances_, 3)})
	importances_feautres = importances_feautres.sort_values('importance', ascending=False).set_index('features')
	importances_feautres = [ls[0] for ls in importances_feautres.values.tolist()]

	with open('F:/django-project/Final_Project/pages/dataset/cocos2d_pickle', 'wb') as cocos2d:
		pickle.dump(clf, cocos2d)

	with open('F:/django-project/Final_Project/pages/dataset/cocos2d_pickle', 'rb') as cocos2d:
		mp = pickle.load(cocos2d)

	score = mp.score(X_test, y_test)
	print("Test score: {0:.2f} %".format(100 * score))
	Ypredict = mp.predict(X_test)
	print(Ypredict[1])

	classification_report = {'accuracy': Accuracy, 'pricision': Precision, 'recall': Recall, 'f1_score': F1_meseaure}

	importance_features = {'importances_feautre': importances_feautres}
	print('-------Importance features----------')
	for k in importance_features:
		Features_list = importance_features[k]
	# for i in Features_list:
	# print(i)
	# print(Features_list)
	print('end')
	plt.barh(features_col, np.round(clf.feature_importances_, 3))
	labels = plt.xlabel('Relative Importance')
	plt.title('Feature Importance of the project cocos2d-x')
	plt.subplots_adjust(left=0.25)
	plt.savefig('F:\django-project\Final_Project\static\media\cocos2d_feature.png')
	# plt.show()

	data = {
		'classification_report': classification_report,
		'importance_feature': Features_list,
		'features': features_col,
		'chart_labels':labels
	}
	return render(request, 'cocos2dPredicting.html', data)


def prdictionResult(request):

	template='PredictionResult.html'

	train = pd.read_csv('F:/django-project/Final_Project/pages/dataset/zendframework.csv')
	features_col = ['First_Status', 'Reputation', 'Changed_file', 'Evaluation_time', 'Nb_Comments',
					'Nb_Commits', 'LC_added', 'LC_deleted']
	X = train[features_col].dropna()
	y = train.classes
	test_size = 0.2  # could also specify train_size=0.7 instead
	# train_size = 0.8
	random_state = 0
	# train_test_split convenience function
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred_prob = clf.predict_proba(X_test)
	# Confusion matrix

	try:
		Accuracy = "{0:.2f}%".format(accuracy_score(y_test, y_pred) * 100)
		Precision = "{0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100)
		Recall = "{0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100)
		F1_meseaure = "{0:.2f}%".format(
			2 * metrics.precision_score(y_test, y_pred) * metrics.recall_score(y_test, y_pred) / (
					metrics.precision_score(y_test, y_pred) + metrics.recall_score(y_test, y_pred)) * 100)
	except ZeroDivisionError:
		print("Error: dividing by zero")
		F1_meseaure = 'nan%'

	importances_feautres = pd.DataFrame({'features': features_col, 'importance': np.round(clf.feature_importances_, 3)})
	importances_feautres = importances_feautres.sort_values('importance', ascending=False).set_index('features')
	importances_feautres = [ls[0] for ls in importances_feautres.values.tolist()]

	with open('F:/django-project/Final_Project/pages/dataset/model_pickle', 'wb') as f:
		pickle.dump(clf, f)

	with open('F:/django-project/Final_Project/pages/dataset/model_pickle', 'rb') as f:
		# prIndex = queryList[i]['pk']
		mp = pickle.load(f)

	score = mp.score(X_test, y_test)
	print("Test score: {0:.2f} %".format(100 * score))
	Ypredict = mp.predict(X_test)


	classification_report = {'accuracy': Accuracy, 'pricision': Precision, 'recall': Recall, 'f1_score': F1_meseaure}

	importance_features = {'importances_feautre': importances_feautres}
	plt.barh(features_col, np.round(clf.feature_importances_, 3))
	labels = plt.xlabel('Relative Importance')
	plt.title('Feature Importance of the project cocos2d-x')
	plt.show()
	data = {
		'classification_report': classification_report,
		'importance_feature': importance_features,
		'features': features_col,
		'chart_labels': labels
	}
	return render(request,template,data)

def angularPrediction(request):
	train = pd.read_csv('F:/django-project/Final_Project/pages/dataset/angularDistributionData.csv')
	features_col = ['First_Status', 'Reputation', 'Changed_file', 'Evaluation_time', 'Nb_Comments',
					'Nb_Commits', 'LC_added', 'LC_deleted']
	X = train[features_col].dropna()
	y = train.classes
	test_size = 0.2  # could also specify train_size=0.7 instead
	# train_size = 0.8
	random_state = 0
	# train_test_split convenience function
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred_prob = clf.predict_proba(X_test)
	# Confusion matrix

	try:
		Accuracy = "{0:.2f}%".format(accuracy_score(y_test, y_pred) * 100)
		Precision = "{0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100)
		Recall = "{0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100)
		F1_meseaure = "{0:.2f}%".format(
			2 * metrics.precision_score(y_test, y_pred) * metrics.recall_score(y_test, y_pred) / (
					metrics.precision_score(y_test, y_pred) + metrics.recall_score(y_test, y_pred)) * 100)
	except ZeroDivisionError:
		print("Error: dividing by zero")
		F1_meseaure = 'nan%'

	importances_feautres = pd.DataFrame({'features': features_col, 'importance': np.round(clf.feature_importances_, 3)})
	importances_feautres = importances_feautres.sort_values('importance', ascending=False).set_index('features')
	importances_feautres = [ls[0] for ls in importances_feautres.values.tolist()]

	with open('F:/django-project/Final_Project/pages/dataset/angular_pickle', 'wb') as angular:
		pickle.dump(clf, angular)

		with open('F:/django-project/Final_Project/pages/dataset/angular_pickle', 'rb') as angular:
			mp = pickle.load(angular)

		score = mp.score(X_test, y_test)
		print("Test score: {0:.2f} %".format(100 * score))
		Ypredict = mp.predict(X_test)
		print(Ypredict[1])

		classification_report = {'accuracy': Accuracy, 'pricision': Precision, 'recall': Recall,
								 'f1_score': F1_meseaure}

		importance_features = {'importances_feautre': importances_feautres}
		plt.barh(features_col, np.round(clf.feature_importances_, 3))
		labels = plt.xlabel('Relative Importance')
		plt.title('Feature Importance of the project Angular')
		plt.subplots_adjust(left=0.25)
		plt.savefig('F:/django-project/Final_Project/static/media/angular_feature.png')
		# plt.show()
		data = {
			'classification_report': classification_report,
			'importance_feature': importance_features,
			'features': features_col,
			'chart_labels': labels
		}
	return render(request, 'angularPredicting.html', data)

def bootstrapPrediction(request):
	train = pd.read_csv('F:/django-project/Final_Project/pages/dataset/twbsDistributionData.csv')
	features_col = ['First_Status', 'Reputation', 'Changed_file', 'Evaluation_time', 'Nb_Comments',
					'Nb_Commits', 'LC_added', 'LC_deleted']
	X = train[features_col].dropna()
	y = train.classes
	test_size = 0.2  # could also specify train_size=0.7 instead
	# train_size = 0.8
	random_state = 0
	# train_test_split convenience function
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred_prob = clf.predict_proba(X_test)
	# Confusion matrix

	try:
		Accuracy = "{0:.2f}%".format(accuracy_score(y_test, y_pred) * 100)
		Precision = "{0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100)
		Recall = "{0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100)
		F1_meseaure = "{0:.2f}%".format(
			2 * metrics.precision_score(y_test, y_pred) * metrics.recall_score(y_test, y_pred) / (
					metrics.precision_score(y_test, y_pred) + metrics.recall_score(y_test, y_pred)) * 100)
	except ZeroDivisionError:
		print("Error: dividing by zero")
		F1_meseaure = 'nan%'

	importances_feautres = pd.DataFrame({'features': features_col, 'importance': np.round(clf.feature_importances_, 3)})
	importances_feautres = importances_feautres.sort_values('importance', ascending=False).set_index('features')
	importances_feautres = [ls[0] for ls in importances_feautres.values.tolist()]

	with open('F:/django-project/Final_Project/pages/dataset/bootstrap_pickle', 'wb') as bootstrap:
		pickle.dump(clf, bootstrap)

	with open('F:/django-project/Final_Project/pages/dataset/bootstrap_pickle', 'rb') as bootstrap:
		mp = pickle.load(bootstrap)

	score = mp.score(X_test, y_test)
	print("Test score: {0:.2f} %".format(100 * score))
	Ypredict = mp.predict(X_test)
	print(Ypredict[1])

	classification_report = {'accuracy': Accuracy, 'pricision': Precision, 'recall': Recall, 'f1_score': F1_meseaure}

	importance_features = {'importances_feautre': importances_feautres}
	plt.barh(features_col, np.round(clf.feature_importances_, 3))
	labels = plt.xlabel('Relative Importance')
	plt.title('Feature Importance of the project Bootstrap')
	plt.subplots_adjust(left=0.25)
	plt.savefig('F:/django-project/Final_Project/static/media/bootstrap_feature.png')
	# plt.show()
	data = {
		'classification_report': classification_report,
		'importance_feature': importance_features,
		'features': features_col,
		'chart_labels': labels
	}
	return render(request, 'bootstrapPredicting.html', data)

def caskroomPrediction(request):
	train = pd.read_csv('F:/django-project/Final_Project/pages/dataset/caskroomDistributionData.csv')
	features_col = ['First_Status', 'Reputation', 'Changed_file', 'Evaluation_time', 'Nb_Comments',
					'Nb_Commits', 'LC_added', 'LC_deleted']
	X = train[features_col].dropna()
	y = train.classes
	test_size = 0.2  # could also specify train_size=0.7 instead
	# train_size = 0.8
	random_state = 0
	# train_test_split convenience function
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred_prob = clf.predict_proba(X_test)
	# Confusion matrix

	try:
		Accuracy = "{0:.2f}%".format(accuracy_score(y_test, y_pred) * 100)
		Precision = "{0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100)
		Recall = "{0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100)
		F1_meseaure = "{0:.2f}%".format(
			2 * metrics.precision_score(y_test, y_pred) * metrics.recall_score(y_test, y_pred) / (
					metrics.precision_score(y_test, y_pred) + metrics.recall_score(y_test, y_pred)) * 100)
	except ZeroDivisionError:
		print("Error: dividing by zero")
		F1_meseaure = 'nan%'

	importances_feautres = pd.DataFrame({'features': features_col, 'importance': np.round(clf.feature_importances_, 3)})
	importances_feautres = importances_feautres.sort_values('importance', ascending=False).set_index('features')
	importances_feautres = [ls[0] for ls in importances_feautres.values.tolist()]

	with open('F:/django-project/Final_Project/pages/dataset/caskroom_pickle', 'wb') as caskroom:
		pickle.dump(clf, caskroom)

	with open('F:/django-project/Final_Project/pages/dataset/bootstrap_pickle', 'rb') as caskroom:
		mp = pickle.load(caskroom)

	score = mp.score(X_test, y_test)
	print("Test score: {0:.2f} %".format(100 * score))
	Ypredict = mp.predict(X_test)
	print(Ypredict[1])

	classification_report = {'accuracy': Accuracy, 'pricision': Precision, 'recall': Recall, 'f1_score': F1_meseaure}

	importance_features = {'importances_feautre': importances_feautres}
	plt.barh(features_col, np.round(clf.feature_importances_, 3))
	labels = plt.xlabel('Relative Importance')
	plt.title('Feature Importance of the project Caskroom')
	plt.subplots_adjust(left=0.25)
	plt.savefig('F:\django-project\Final_Project\static\media\caskroom_feature.png')
	# plt.show()
	data = {
		'classification_report': classification_report,
		'importance_feature': importance_features,
		'features': features_col,
		'chart_labels': labels
	}
	return render(request, 'caskroomPredicting.html', data)

def zendframeworkPrediction(request):
	train = pd.read_csv('F:/django-project/Final_Project/pages/dataset/zendframework.csv')
	features_col = ['First_Status', 'Reputation', 'Changed_file', 'Evaluation_time', 'Nb_Comments',
					'Nb_Commits', 'LC_added', 'LC_deleted']
	X = train[features_col].dropna()
	y = train.classes
	test_size = 0.2  # could also specify train_size=0.7 instead
	# train_size = 0.8
	random_state = 0
	# train_test_split convenience function
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred_prob = clf.predict_proba(X_test)
	# Confusion matrix

	try:
		Accuracy = "{0:.2f}%".format(accuracy_score(y_test, y_pred) * 100)
		Precision = "{0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100)
		Recall = "{0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100)
		F1_meseaure = "{0:.2f}%".format(
			2 * metrics.precision_score(y_test, y_pred) * metrics.recall_score(y_test, y_pred) / (
					metrics.precision_score(y_test, y_pred) + metrics.recall_score(y_test, y_pred)) * 100)
	except ZeroDivisionError:
		print("Error: dividing by zero")
		F1_meseaure = 'nan%'

	importances_feautres = pd.DataFrame({'features': features_col, 'importance': np.round(clf.feature_importances_, 3)})
	importances_feautres = importances_feautres.sort_values('importance', ascending=False).set_index('features')
	importances_feautres = [ls[0] for ls in importances_feautres.values.tolist()]

	with open('F:/django-project/Final_Project/pages/dataset/zendframework_pickle', 'wb') as zendframework:
		pickle.dump(clf, zendframework)

	with open('F:/django-project/Final_Project/pages/dataset/zendframework_pickle', 'rb') as zendframework:
		mp = pickle.load(zendframework)

	score = mp.score(X_test, y_test)
	print("Test score: {0:.2f} %".format(100 * score))
	Ypredict = mp.predict(X_test)
	print(Ypredict[1])

	classification_report = {'accuracy': Accuracy, 'pricision': Precision, 'recall': Recall, 'f1_score': F1_meseaure}

	importance_features = {'importances_feautre': importances_feautres}

	importance_features = {'importances_feautre': importances_feautres}
	plt.barh(features_col, np.round(clf.feature_importances_, 3))
	labels = plt.xlabel('Relative Importance')
	plt.title('Feature Importance of the project Zendframework')
	plt.subplots_adjust(left=0.25)
	plt.savefig('F:\django-project\Final_Project\static\media\zendframework_feature.png')
	# plt.show()
	data = {
		'classification_report': classification_report,
		'importance_feature': importance_features,
		'features': features_col,
		'chart_labels': labels
	}
	return render(request, 'zendframework.html', data)



def get_basicInformation(request):
	return render(request, 'basic_resutls.html',{})

def get_status(request):
	return render(request, 'status.html',{})

def changed_files_view(request):
	return render(request, 'changed_files.html',{})

def percent_merged_view(request):
	return render(request, 'mergre_percent.html',{})

def nonreop_reope_view(request):
	return render(request, 'comparison_reop_nonreop.html',{})


def reop_reasons_view(request):
	return render(request,'reop_reasons.html', {})

def impact_view(request):
	return render(request,'impacts.html', {})








# from .filters import PullRequestsFilter
# from django.views.generic import ListView, DetailView
#
# class PullRequestlistView(ListView):
# 	model=Pull_Requests
# 	templatename='pr_filter_list.html'
#
#
#
# 	def get_context_data(self, **kwargs):
# 		context=super().get_context_data(**kwargs)
# 		context['filter']=Pull_Requests(self.request.GET, queryset=self.get_queryset())
# 		# pull_request_list=Pull_Requests.objects.all()
# 		# pr_filter=PullRequestsFilter(self.request.GET, queryset=pull_request_list)
#
# 		return context


# #---------------------------------Stop this first-------------------------------
# def upload_file(request):
# 	with open('RailsDistributionData.csv', 'r') as csvfile:
# 		#reader = csv.reader(csvfile)
# 		reader=csv.DictReader(csvfile)
# 		for row in reader:
# 			PR_Project = row['foreign_key']
# 			PR_ID = row['pr_id']
# 			Nb_comments = row['Num_Comments_before_Closed']
# 			LC_added = row['Num_lines_added']
# 			LC_deleted = row['Num_lines_deleted']
# 			Nb_commits = row['Num_commits_before_Closed']
# 			Nb_changed_fies = row['Changed_file']
# 			First_Closed_status = row['FirstStatus']
#			Evaluation_time = row['Evaluation_time']
# 			Reputation = row['Reputation']
# 			Predicted_class = row['Label']
# 			new_data = Pull_Requests(pr_project=PR_Project, pr_id=PR_ID, nb_comments=Nb_comments,
# 									 nb_added_lines_code=LC_added,
# 									 nb_deleted_lines_code=LC_deleted, nb_commits=Nb_commits,
# 									 nb_changed_fies=Nb_changed_fies,
# 									 Closed_status=First_Closed_status, reputation=Reputation, Label=Predicted_class)
# 			new_data.save()
# 	return render(request, 'add_pull_requests.html')


def IndexSymfony(request):
    limit=12
    pull_requestsList = symfony_Project.objects.all()
    paginator = Paginator(pull_requestsList, limit)
    page=request.GET.get('page')
    try:
        pullrequests=paginator.page(page)
    except PageNotAnInteger:
        pullrequests=paginator.page(1)
    except EmptyPage:
        pullrequests=paginator.page(paginator.num_pages)
    return render(request, 'symfonyIndex.html', {'pullrequests': pullrequests, 'pull_requestsList':pull_requestsList})


def SymfonyPredictionResult(request):
	pull_requestsList = symfony_Project.objects.all()
	project = request.GET.get('project')
	prId = request.GET.get('id')
	newPredictionData = {}

	if pull_requestsList.filter(pr_project=project):
		# indexId=pull_requestsList.filter(pr_project=project).valuest('pr_project')
		q = pull_requestsList.filter(pr_id=prId).values('pk', 'pr_project', 'pr_id', 'nb_comments','nb_added_lines_code', 'nb_deleted_lines_code', 'nb_commits',
														'nb_changed_fies', 'time_evaluation', 'Closed_status','reputation', 'Label')

	# queryList = list(q)

	dict = q[0]
	print(dict)
	print('----------------------')
	print(dict['pk'])

	prIndex = dict['pk']

	newPredictionData = {'index': dict['pk'], 'project': dict['pr_project'], 'prId': dict['pr_id'],
						 'comments': dict['nb_comments'], 'lc_added': dict['nb_added_lines_code'],
						 'lc_deleted': dict['nb_deleted_lines_code'], 'commits': dict['nb_commits'],
						 'changed_files': dict['nb_changed_fies'],
						 'evaluation_time': dict['time_evaluation'], 'firstStatus': dict['Closed_status'],
						 'reputation': dict['reputation'],
						 'classLabel': dict['Label']}
	data_list = list(newPredictionData.values())
	print('-----------------')
	print(data_list)
	print('-----------------')
	for i in range(len(data_list)):
		if data_list[i] == "Reopened":
			data_list[i] = 1
		elif data_list[i] == "Non-Reopened":
			data_list[i] = 0
		elif data_list[i] == "Rejected":
			data_list[i] = 0
		elif data_list[i] == "Accepted":
			data_list[i] = 1
		else:
			pass
	# y = data[-1]

	for j in range(len(data_list)):
		newData = data_list[0:j]
	# print('-----------------')
	# print('New data', newData)
	train = pd.read_csv('F:/django-project/Final_Project/pages/dataset/symfonyDistributionData.csv')
	features_col = ['First_Status', 'Reputation', 'Changed_file', 'Evaluation_time', 'Nb_Comments',
					'Nb_Commits', 'LC_added', 'LC_deleted']
	X = train[features_col].dropna()
	y = train.classes
	test_size = 0.2  # could also specify train_size=0.7 instead
	# train_size = 0.8
	random_state = 0
	# train_test_split convenience function
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred_prob = clf.predict_proba(X_test)
	# Confusion matrix

	try:
		Accuracy = "{0:.2f}%".format(accuracy_score(y_test, y_pred) * 100)
		Precision = "{0:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100)
		Recall = "{0:.2f}%".format(metrics.recall_score(y_test, y_pred) * 100)
		F1_meseaure = "{0:.2f}%".format(
			2 * metrics.precision_score(y_test, y_pred) * metrics.recall_score(y_test, y_pred) / (
					metrics.precision_score(y_test, y_pred) + metrics.recall_score(y_test, y_pred)) * 100)
	# F1_meseaure=("{0:.2f}%".format(2*Precision*Recall)/(Precision+Recall))*100
	except ZeroDivisionError:
		print("Error: dividing by zero")
		F1_meseaure = 'nan%'

	importances_feautres = pd.DataFrame({'features': features_col, 'importance': np.round(clf.feature_importances_, 3)})
	importances_feautres = importances_feautres.sort_values('importance', ascending=False).set_index('features')
	importances_feautres = [ls[0] for ls in importances_feautres.values.tolist()]

	with open('F:/django-project/Final_Project/pages/dataset/symfony_pickle', 'wb') as symfony:
		pickle.dump(clf, symfony)

	with open('F:/django-project/Final_Project/pages/dataset/symfony_pickle', 'rb') as symfony:
		# prIndex = queryList[i]['pk']
		mp = pickle.load(symfony)

	score = mp.score(X_test, y_test)
	print("Test score: {0:.2f} %".format(100 * score))
	Ypredict = mp.predict(X_test)
	print(Ypredict[1])
	mp.predict(X[0:prIndex+5])
	predict=y[1]
	print(predict)


	classification_report = {'accuracy': Accuracy, 'pricision': Precision, 'recall': Recall, 'f1_score': F1_meseaure}

	importance_features = {'importances_feautre': importances_feautres}

	data = {
		'new_data': newPredictionData,
		'classification_report': classification_report,
		'importance_feature': importance_features,
		'features': features_col,
		# 'projects':projects'
		'predicted':predict
	}
	return render(request, 'SymfonyPredResult.html', data)
