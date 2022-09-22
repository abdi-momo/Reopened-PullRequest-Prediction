import csv
import os
#import django
from pages.models import Pull_Requests
# path='C:/Users/ABDILLAH/Desktop/Prototype_DataSet'
# os.chdir(path)
with open('RailsDistributionData.csv', 'r') as csvfile:
    #reader = csv.reader(csvfile)
    reader=csv.DictReader(csvfile)
    #print (type(reader))
    for row in reader:
        PR_Project=row['foreign_key']
        PR_ID=row['pr_id']
        Nb_comments=row['Num_Comments_before_Closed']
        LC_added=row['Num_lines_added']
        LC_deleted=row['Num_lines_deleted']
        Nb_commits=row['Num_commits_before_Closed']
        Nb_changed_fies=row['Changed_file']
        First_Closed_status=row['FirstStatus']
        Reputation=row['Reputation']
        Predicted_class=row['Label']
        new_data=Pull_Requests(pr_project=PR_Project, pr_id=PR_ID, nd_comments=Nb_comments, nb_added_lines_code=LC_added,
                               nb_deleted_lines_code=LC_deleted,nb_commits=Nb_commits, nb_changed_fies=Nb_changed_fies,
                               Closed_status=First_Closed_status, reputation=Reputation, Label=Predicted_class)
        new_data.save()
        #project=row[0])
        #p=Pull_Requests(pr_id=row[1], nd_comments=row[7], nb_added_lines_code=row[9], nb_deleted_lines_code=row[10],
        #nb_commits=row[8], nb_changed_fies=row[5], Closed_status=row[2], reputation=row[4], Label=row[11], project=row[0])
        # p.save()


