from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import Pull_Requests, Projects, rails_Project,angular_Project,zendframework_Project

class ProfileInline(admin.StackedInline):
    # model = UserProfile
    can_delete = False
    # verbose_name_plural = 'UserProfile'
    fk_name = 'user'
#
class CustomUserAdmin(UserAdmin):
    inlines = (ProfileInline, )

    def get_inline_instances(self, request, obj=None):
        if not obj:
            return list()
        return super(CustomUserAdmin, self).get_inline_instances(request, obj)

admin.site.register(Projects)
admin.site.register(Pull_Requests)
admin.site.unregister(User)
admin.site.register(rails_Project)
# admin.site.register(User, CustomUserAdmin)
admin.site.register(zendframework_Project)
admin.site.register(angular_Project)
