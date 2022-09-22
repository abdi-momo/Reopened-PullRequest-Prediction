from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

#project=models.ForeignKey(Projects, on_delete=models.CASCADE)
#create_date=models.DateTimeField('Date submitted')
#contributor_name=models.CharField(max_length=30)
#first_close_time=models.DateTimeField('First closing date')

class Projects(models.Model):
	project_name=models.CharField(max_length=30, unique=True)
	project_owner=models.CharField(max_length=35)

class Pull_Requests(models.Model):
	pr_project=models.CharField(max_length=45)
	pr_id=models.CharField(max_length=10)
	nb_comments=models.IntegerField(default=0)
	nb_added_lines_code=models.IntegerField(default=0)
	nb_deleted_lines_code=models.IntegerField(default=0)
	nb_commits=models.IntegerField(default=0)
	nb_changed_fies=models.IntegerField(default=0)
	time_evaluation=models.FloatField(default=0.00)
	Closed_status=models.CharField(max_length=15)
	reputation=models.FloatField(default=0.00)
	#pred_class=((True,'Reopened'), (False,'Non-reopened'))
	Label=models.CharField(max_length=30, default="Non-reopened")
	#project=models.ForeignKey(Projects, on_delete=models.CASCADE)
	def __str__(self):
		return '{} {} {} {} {} {} {} {} {} {} {}'.format(self.pr_project, self.pr_id, self.nb_comments, self.nb_added_lines_code, self.nb_deleted_lines_code, \
			   self.nb_commits, self.nb_changed_fies, self.time_evaluation, self.Closed_status, self.reputation, self.Label)

class rails_Project(models.Model):
	pr_project = models.CharField(max_length=20)
	pr_id = models.CharField(max_length=10)
	nb_comments = models.IntegerField(default=0)
	nb_added_lines_code = models.IntegerField(default=0)
	nb_deleted_lines_code = models.IntegerField(default=0)
	nb_commits = models.IntegerField(default=0)
	nb_changed_fies = models.IntegerField(default=0)
	time_evaluation = models.FloatField(default=0.00)
	Closed_status = models.CharField(max_length=15)
	reputation = models.FloatField(default=0.00)
	# pred_class=((True,'Reopened'), (False,'Non-reopened'))
	Label = models.CharField(max_length=30, default="Non-reopened")
	def __str__(self):
		return '{} {} {} {} {} {} {} {} {} {}'.format(self.pr_project,self.pr_id, self.nb_comments, self.nb_added_lines_code, self.nb_deleted_lines_code, \
			   self.nb_commits, self.nb_changed_fies, self.time_evaluation, self.Closed_status, self.reputation, self.Label)

class cocos2d_Project(models.Model):
	pr_project = models.CharField(max_length=20)
	pr_id = models.CharField(max_length=10)
	nb_comments = models.IntegerField(default=0)
	nb_added_lines_code = models.IntegerField(default=0)
	nb_deleted_lines_code = models.IntegerField(default=0)
	nb_commits = models.IntegerField(default=0)
	nb_changed_fies = models.IntegerField(default=0)
	time_evaluation = models.FloatField(default=0.00)
	Closed_status = models.CharField(max_length=15)
	reputation = models.FloatField(default=0.00)
	# pred_class=((True,'Reopened'), (False,'Non-reopened'))
	Label = models.CharField(max_length=30, default="Non-reopened")
	def __str__(self):
		return '{} {} {} {} {} {} {} {} {} {}'.format(self.pr_project,self.pr_id, self.nb_comments, self.nb_added_lines_code, self.nb_deleted_lines_code, \
			   self.nb_commits, self.nb_changed_fies, self.time_evaluation, self.Closed_status, self.reputation, self.Label)

class symfony_Project(models.Model):
	pr_project = models.CharField(max_length=20)
	pr_id = models.CharField(max_length=10)
	nb_comments = models.IntegerField(default=0)
	nb_added_lines_code = models.IntegerField(default=0)
	nb_deleted_lines_code = models.IntegerField(default=0)
	nb_commits = models.IntegerField(default=0)
	nb_changed_fies = models.IntegerField(default=0)
	time_evaluation = models.FloatField(default=0.00)
	Closed_status = models.CharField(max_length=15)
	reputation = models.FloatField(default=0.00)
	# pred_class=((True,'Reopened'), (False,'Non-reopened'))
	Label = models.CharField(max_length=30, default="Non-reopened")
	def __str__(self):
		return '{} {} {} {} {} {} {} {} {} {}'.format(self.pr_project,self.pr_id, self.nb_comments, self.nb_added_lines_code, self.nb_deleted_lines_code, \
			   self.nb_commits, self.nb_changed_fies, self.time_evaluation, self.Closed_status, self.reputation, self.Label)

class caskroom_Project(models.Model):
	pr_project = models.CharField(max_length=20)
	pr_id = models.CharField(max_length=10)
	nb_comments = models.IntegerField(default=0)
	nb_added_lines_code = models.IntegerField(default=0)
	nb_deleted_lines_code = models.IntegerField(default=0)
	nb_commits = models.IntegerField(default=0)
	nb_changed_fies = models.IntegerField(default=0)
	time_evaluation = models.FloatField(default=0.00)
	Closed_status = models.CharField(max_length=15)
	reputation = models.FloatField(default=0.00)
	# pred_class=((True,'Reopened'), (False,'Non-reopened'))
	Label = models.CharField(max_length=30, default="Non-reopened")
	def __str__(self):
		return '{} {} {} {} {} {} {} {} {} {}'.format(self.pr_project,self.pr_id, self.nb_comments, self.nb_added_lines_code, self.nb_deleted_lines_code, \
			   self.nb_commits, self.nb_changed_fies, self.time_evaluation, self.Closed_status, self.reputation, self.Label)


class zendframework_Project(models.Model):
	pr_project = models.CharField(max_length=20)
	pr_id = models.CharField(max_length=10)
	nb_comments = models.IntegerField(default=0)
	nb_added_lines_code = models.IntegerField(default=0)
	nb_deleted_lines_code = models.IntegerField(default=0)
	nb_commits = models.IntegerField(default=0)
	nb_changed_fies = models.IntegerField(default=0)
	time_evaluation = models.FloatField(default=0.00)
	Closed_status = models.CharField(max_length=15)
	reputation = models.FloatField(default=0.00)
	# pred_class=((True,'Reopened'), (False,'Non-reopened'))
	Label = models.CharField(max_length=30, default="Non-reopened")
	def __str__(self):
		return '{} {} {} {} {} {} {} {} {} {}'.format(self.pr_project,self.pr_id, self.nb_comments, self.nb_added_lines_code, self.nb_deleted_lines_code, \
			   self.nb_commits, self.nb_changed_fies, self.time_evaluation, self.Closed_status, self.reputation, self.Label)


class angular_Project(models.Model):
	pr_project = models.CharField(max_length=20)
	pr_id = models.CharField(max_length=10)
	nb_comments = models.IntegerField(default=0)
	nb_added_lines_code = models.IntegerField(default=0)
	nb_deleted_lines_code = models.IntegerField(default=0)
	nb_commits = models.IntegerField(default=0)
	nb_changed_fies = models.IntegerField(default=0)
	time_evaluation = models.FloatField(default=0.00)
	Closed_status = models.CharField(max_length=15)
	reputation = models.FloatField(default=0.00)
	# pred_class=((True,'Reopened'), (False,'Non-reopened'))
	Label = models.CharField(max_length=30, default="Non-reopened")
	def __str__(self):
		return '{} {} {} {} {} {} {} {} {} {}'.format(self.pr_project,self.pr_id, self.nb_comments, self.nb_added_lines_code, self.nb_deleted_lines_code, \
			   self.nb_commits, self.nb_changed_fies, self.time_evaluation, self.Closed_status, self.reputation, self.Label)


class bootstrap_Project(models.Model):
	pr_project = models.CharField(max_length=20)
	pr_id = models.CharField(max_length=10)
	nb_comments = models.IntegerField(default=0)
	nb_added_lines_code = models.IntegerField(default=0)
	nb_deleted_lines_code = models.IntegerField(default=0)
	nb_commits = models.IntegerField(default=0)
	nb_changed_fies = models.IntegerField(default=0)
	time_evaluation = models.FloatField(default=0.00)
	Closed_status = models.CharField(max_length=15)
	reputation = models.FloatField(default=0.00)
	# pred_class=((True,'Reopened'), (False,'Non-reopened'))
	Label = models.CharField(max_length=30, default="Non-reopened")
	def __str__(self):
		return '{} {} {} {} {} {} {} {} {} {}'.format(self.pr_project,self.pr_id, self.nb_comments, self.nb_added_lines_code, self.nb_deleted_lines_code, \
			   self.nb_commits, self.nb_changed_fies, self.time_evaluation, self.Closed_status, self.reputation, self.Label)

# class UserProfile(models.Model):
# 	STUDENT = 1
# 	TEACHER = 2
# 	SUPERVISOR = 3
# 	ROLE_CHOICES = (
# 		(STUDENT, 'Student'),
# 		(TEACHER, 'Teacher'),
# 		(SUPERVISOR, 'Supervisor'),
# 	)
# 	user=models.OneToOneField(User, on_delete=models.CASCADE)
# 	username = models.CharField(max_length=30, blank=True)
# 	email = models.EmailField()
# 	email2=models.EmailField()
# 	password=models.CharField(max_length=50)
# 	role = models.PositiveSmallIntegerField(choices=ROLE_CHOICES, null=True, blank=True)
#
# 	def __str__(self):  # __unicode__ for Python 2
# 		return self.user.username

