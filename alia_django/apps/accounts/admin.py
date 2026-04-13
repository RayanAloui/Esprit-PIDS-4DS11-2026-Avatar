from django.contrib import admin
from .models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display  = ('user', 'role', 'region', 'telephone', 'created_at')
    list_filter   = ('role', 'region')
    search_fields = ('user__username', 'user__first_name',
                     'user__last_name', 'user__email')
    raw_id_fields = ('user',)
