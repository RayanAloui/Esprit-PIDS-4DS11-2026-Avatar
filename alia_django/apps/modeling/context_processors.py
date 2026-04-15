from django.urls import NoReverseMatch, reverse


def modeling_nav(_request):
    try:
        return {"modeling_url": reverse("modeling_index")}
    except NoReverseMatch:
        return {"modeling_url": "/alia-api/"}
