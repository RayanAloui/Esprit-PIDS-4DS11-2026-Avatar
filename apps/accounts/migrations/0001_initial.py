from django.conf import settings
from django.db   import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='UserProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True,
                                           serialize=False, verbose_name='ID')),
                ('role', models.CharField(
                    choices=[('delegue', 'Délégué médical'), ('manager', 'Manager')],
                    default='delegue', max_length=20)),
                ('region',    models.CharField(blank=True, default='Grand Tunis', max_length=100)),
                ('telephone', models.CharField(blank=True, max_length=20)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.OneToOneField(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='profile',
                    to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name':        'Profil utilisateur',
                'verbose_name_plural': 'Profils utilisateurs',
            },
        ),
    ]
