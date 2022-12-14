from django import forms
from django.contrib.auth.models import User
from .models import  Pull_Requests, Projects
from django.contrib.auth import (authenticate,get_user_model, login, logout)

User=get_user_model()
class UserForm(forms.Form):
    username=forms.CharField()
    password=forms.CharField(widget=forms.PasswordInput)

    def clean(self, *args, **kwargs):
        username=self.cleaned_data.get("username")
        password = self.cleaned_data.get("password")
        user=authenticate(username=username, password=password)
        # user_qs=User.objects.filter(username=username)
        # if user_qs.count()==1:
        #     user=user_qs.first()
        if username and password:
            user = authenticate(username=username, password=password)
            if not user:
                raise forms.ValidationError("This user des not exist")
            if not user.check_password(password):
                raise forms.ValidationError("Incorrect password")
            if not user.is_active:
                raise forms.ValidationError("This user is no longer active. ")
        return super(UserForm, self).clean(*args, **kwargs)

    class Meta:
        model=User
        fields=('username', 'email', 'password')


class UserProfileForm(forms.ModelForm):
    email = forms.EmailField(label='Email Address')
    email2=forms.EmailField(label='Confirm Email')
    password = forms.CharField(widget=forms.PasswordInput)
    class Meta:
        model=User
        fields=['username',
                'email',
                'email2',
                'password'
                ]

    # def clean(self, *args, **kwargs):
    #     email = self.cleaned_data.get('email')
    #     email2 = self.cleaned_data.get('email2')
    #     if email != email2:
    #         raise forms.ValidationError("Emails must much")
    #     email_qs = User.objects.filter(email=email)
    #     if email_qs.exists():
    #         raise forms.ValidationError("This email has already been registered")
    #     return super(UserProfileForm, self).clean(*args, **kwargs)

    def clean_email2(self):
        email = self.cleaned_data.get('email')
        email2 = self.cleaned_data.get('email2')
        if email != email2:
            raise forms.ValidationError("Emails must much")
        email_qs=User.objects.filter(email=email)
        if email_qs.exists():
            raise forms.ValidationError("This email has already been registered")
        return email

