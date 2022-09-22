from django.contrib.auth.models import User
from.models import Pull_Requests
import django_filters
class PullRequestsFilter(django_filters.FilterSet):
    class Meta:
        mooel=Pull_Requests
        fields=['pr_project', 'pr_project', 'nb_comments', 'nb_added_lines_code', 'nb_deleted_lines_code', 'nb_commits',
                'nb_changed_fies', 'time_evaluation', 'Closed_status', 'reputation', 'Label']
