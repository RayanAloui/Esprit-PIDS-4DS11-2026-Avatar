from django.db import models
from django.contrib.auth.models import User

class SimulationHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='simulations')
    date = models.DateTimeField(auto_now_add=True)
    product_id = models.CharField(max_length=100)
    interlocutor_id = models.CharField(max_length=100)
    score_global = models.FloatField(default=0)
    xp_earned = models.IntegerField(default=0)
    is_success = models.BooleanField(default=False)
