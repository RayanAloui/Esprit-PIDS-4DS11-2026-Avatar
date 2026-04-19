"""
Modèles SQLite — Historique des analyses NLP
"""
from django.db import models
from django.utils import timezone
from django.conf import settings

class NLPAnalysis(models.Model):
    """
    Stocke chaque analyse NLP effectuée via la page Avatar et Simulateur.
    Permet de consulter l'historique des évaluations.
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True, verbose_name="Délégué")

    # ── Entrée ────────────────────────────────────────────────────────
    objection      = models.TextField(verbose_name="Objection du médecin")
    response       = models.TextField(verbose_name="Réponse du délégué")
    niveau_alia_input = models.CharField(
        max_length=20, default="Junior",
        verbose_name="Niveau ALIA saisi",
        choices=[
            ('Débutant', 'Débutant'),
            ('Junior',   'Junior'),
            ('Confirmé', 'Confirmé'),
            ('Expert',   'Expert'),
        ]
    )

    # ── Résultats T1 — Qualité ────────────────────────────────────────
    quality        = models.CharField(max_length=20, verbose_name="Qualité")
    overall_score  = models.FloatField(verbose_name="Score global ALIA")

    # ── Résultats T2 — Scores détaillés ──────────────────────────────
    score_scientific = models.FloatField(verbose_name="Score scientifique")
    score_clarity    = models.FloatField(verbose_name="Score clarté")
    score_objection  = models.FloatField(verbose_name="Score objection")

    # ── Résultats T3/T4/T5/T6 ────────────────────────────────────────
    sentiment        = models.CharField(max_length=20, verbose_name="Sentiment")
    niveau_alia_pred = models.CharField(max_length=20, verbose_name="Niveau ALIA prédit")
    visit_format     = models.CharField(max_length=20, verbose_name="Format visite")
    conformite       = models.BooleanField(verbose_name="Conformité")
    acrv_score       = models.IntegerField(default=0, verbose_name="Score A-C-R-V")

    # ── Meta ──────────────────────────────────────────────────────────
    created_at     = models.DateTimeField(default=timezone.now, verbose_name="Date analyse")

    class Meta:
        verbose_name        = "Analyse NLP"
        verbose_name_plural = "Analyses NLP"
        ordering            = ['-created_at']

    def __str__(self):
        return f"[{self.quality}] {self.overall_score:.1f}/10 — {self.created_at.strftime('%d/%m/%Y %H:%M')}"

    @property
    def quality_color(self):
        return {'Excellent': '#2ecc71', 'Bon': '#f39c12', 'Faible': '#e74c3c'}.get(self.quality, '#888')
