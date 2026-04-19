"""
models.py — accounts
=====================
Profil utilisateur ALIA étendu via OneToOne sur User Django.
Deux rôles : délégué (accès simulation + analytics perso)
             manager (accès analytics global + tous les délégués)
"""
from django.db   import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    ROLE_CHOICES = [
        ('delegue', 'Délégué médical'),
        ('manager', 'Manager'),
    ]

    user        = models.OneToOneField(User, on_delete=models.CASCADE,
                                       related_name='profile')
    role        = models.CharField(max_length=20, choices=ROLE_CHOICES,
                                   default='delegue')
    region      = models.CharField(max_length=100, blank=True,
                                   default='Grand Tunis')
    telephone   = models.CharField(max_length=20, blank=True)
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name        = 'Profil utilisateur'
        verbose_name_plural = 'Profils utilisateurs'

    def __str__(self):
        return f"{self.user.get_full_name() or self.user.username} ({self.get_role_display()})"

    @property
    def display_name(self):
        return self.user.get_full_name() or self.user.username

    @property
    def initials(self):
        """2 lettres pour l'avatar dans la navbar."""
        fn = self.user.first_name
        ln = self.user.last_name
        if fn and ln:
            return (fn[0] + ln[0]).upper()
        return self.user.username[:2].upper()

    @property
    def is_manager(self):
        return self.role == 'manager'
