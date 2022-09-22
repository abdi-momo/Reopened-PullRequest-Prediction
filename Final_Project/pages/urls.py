from django.urls import  path
from django.conf.urls import url
from . import views
from .views import upload_file
from django.contrib import admin

from pages.views import (login_view, register_view, logout_view)


urlpatterns = [
        path('', views.login_view, name='login_view'),
        # url(r'^$', views.home),
        # path('', views.home, name='home'),
        path('index/', views.index, name='index'),
        path('RailsIndex/',views.RailsIndex, name='RailsIndex'),
        # path('register/', views.register, name='register'),
        path('add_pull_requests/', views.pull_request_upload, name='add_pull_requests'),
        path('upload_file/', views.upload_file, name='upload_file'),

        path('resultOfPredicting/', views.resultOfPredicting, name='resultOfPredicting'),

        url(r'login/', login_view, name='login'),
        url(r'logout/', logout_view, name='logout'),
        url(r'register/', register_view, name='register'),


        path('basic_resuls', views.get_basicInformation, name='basic_resuls'),
        path('status', views.get_status, name='status'),
        path('changed_files', views.changed_files_view, name='changed_files'),
        path('mergre_percent', views.percent_merged_view, name='mergre_percent'),
        path('comparison_reop_nonreop', views.nonreop_reope_view, name='comparison_reop_nonreop'),
        path('reo_reasons', views.reop_reasons_view, name='reo_reasons'),
        path('impact', views.impact_view, name='impact'),
        path('predictionResult',  views.prdictionResult, name='predictionResult'),
        path('rails',  views.railsPrediction, name='rails'),
        path('symfony',  views.symfonyPrediction, name='symfony'),
        path('cocosd',  views.cocos2dPrediction, name='cocosd'),
        path('angular',  views.angularPrediction, name='angular'),
        path('bootstrap',  views.bootstrapPrediction, name='bootstrap'),
        path('caskroom',  views.caskroomPrediction, name='caskroom'),
        path('zendframework/', views.zendframeworkPrediction, name='zendframework'),
        path('railsProject/', views.railsProject, name='railsProject'),
        path('cocosUpload/', views.cocos2dProject_Upload, name='cocosUpload'),
        path('symfonyUpload/', views.SymfonyProject_Upload, name='symfonyUpload'),
        path('caskroomUpload/', views.CaskroomProject_Upload, name='caskroomUpload'),
        path('zendframeworkUpload/', views.Zendframework_Upload, name='zendframeworkUpload'),
        path('angularUpload/', views.Angular_Upload, name='angularUpload'),
        path('bootstrapUpload/', views.Bootstrap_Upload, name='bootstrapUpload'),


        path('indexSymfony/',views.IndexSymfony,name='indexSymfony'),



        path('SymfonyPredResult/', views.SymfonyPredictionResult, name='symfonyPredictResult'),


	]
