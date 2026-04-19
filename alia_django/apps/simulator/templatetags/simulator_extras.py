from django import template

register = template.Library()


@register.filter(name="chr")
def chr_filter(value):
    """Return the Unicode character for an integer code point (e.g. 65 -> 'A')."""
    try:
        return chr(int(value))
    except (TypeError, ValueError):
        return ""
