from django.contrib import admin
from django.urls import include, path
from  pages import views
from django.views.generic.base import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('pages.urls')),
    # path('accounts/', include('django.contrib.auth.urls')),
    path('', TemplateView.as_view(template_name='register.html'), name='login'),  # new
]
