"""
views.py — accounts
====================
Login / Logout / Profil utilisateur ALIA.
"""
from django.shortcuts               import render, redirect
from django.contrib.auth            import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib                 import messages
from django.views.decorators.http   import require_http_methods

from .models import UserProfile


# ── Login ──────────────────────────────────────────────────────────────

def login_view(request):
    """GET + POST — Page de connexion ALIA."""
    # Déjà connecté → accueil
    if request.user.is_authenticated:
        return redirect('home:index')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        if not username or not password:
            messages.error(request, 'Identifiant et mot de passe requis.')
            return render(request, 'accounts/login.html', {'next': request.POST.get('next', '/')})

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            # S'assurer que le profil existe (création lazy)
            UserProfile.objects.get_or_create(user=user)
            next_url = request.POST.get('next') or request.GET.get('next') or '/'
            # Sécurité : ignorer les next absolus externes
            if not next_url.startswith('/'):
                next_url = '/'
            return redirect(next_url)
        else:
            messages.error(request, 'Identifiant ou mot de passe incorrect.')

    return render(request, 'accounts/login.html', {
        'next': request.GET.get('next', '/'),
    })


# ── Logout ─────────────────────────────────────────────────────────────

@require_http_methods(["POST"])
def logout_view(request):
    """POST — Déconnexion sécurisée (CSRF protégé)."""
    logout(request)
    return redirect('accounts:login')


# ── Profil ─────────────────────────────────────────────────────────────

@login_required
def profile_view(request):
    """GET + POST — Page de profil et modification des infos."""
    profile, _ = UserProfile.objects.get_or_create(user=request.user)

    if request.method == 'POST':
        first_name = request.POST.get('first_name', '').strip()
        last_name  = request.POST.get('last_name', '').strip()
        email      = request.POST.get('email', '').strip()
        region     = request.POST.get('region', '').strip()
        telephone  = request.POST.get('telephone', '').strip()

        request.user.first_name = first_name
        request.user.last_name  = last_name
        request.user.email      = email
        request.user.save(update_fields=['first_name', 'last_name', 'email'])

        profile.region    = region
        profile.telephone = telephone
        profile.save(update_fields=['region', 'telephone'])

        messages.success(request, 'Profil mis à jour avec succès.')
        return redirect('accounts:profile')

    return render(request, 'accounts/profile.html', {
        'page'   : 'profile',
        'profile': profile,
    })
