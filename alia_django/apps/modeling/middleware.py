from time import perf_counter

from django.utils.deprecation import MiddlewareMixin

from apps.modeling.metrics import log_request_metrics


class RequestMetricsMiddleware(MiddlewareMixin):
    """Middleware that logs response latency and process CPU/memory metrics."""

    def process_request(self, request):
        request._metrics_start = perf_counter()

    def process_response(self, request, response):
        start = getattr(request, "_metrics_start", None)
        if start is not None:
            latency_s = perf_counter() - start
            log_request_metrics(request.path, request.method, latency_s, getattr(response, "status_code", 0))
        return response
