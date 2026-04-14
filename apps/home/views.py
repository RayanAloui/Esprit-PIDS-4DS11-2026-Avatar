from django.shortcuts import render

def home_index(request):
    context = {
        'page': 'home',
        'stats': {
            'pharmacies': 64,
            'produits'  : 16,
            'niveaux'   : 4,
            'objections': 35,
        }
    }
    return render(request, 'home/index.html', context)
