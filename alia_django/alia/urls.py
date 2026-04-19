from django.contrib import admin
from django.urls    import path, include
from django.conf    import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/',       admin.site.urls),
    path('accounts/',    include('apps.accounts.urls')),   # ← Auth ALIA
    path('',             include('apps.modeling.urls')),
    path('',             include('apps.home.urls')),
    path('avatar/',      include('apps.avatar.urls')),
    path('routes/',      include('apps.routes.urls')),
    path('analytics/',   include('apps.analytics.urls')),
    path('simulator/',   include('apps.simulator.urls')),
    path('crm/',         include('apps.crm.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
