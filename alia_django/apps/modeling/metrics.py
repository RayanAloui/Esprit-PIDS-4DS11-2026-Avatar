import logging
import os
from typing import Dict

log = logging.getLogger("alia.metrics")

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import GPUtil
except ImportError:
    GPUtil = None


def get_system_metrics() -> Dict[str, float]:
    """Return current process/system CPU, memory, and GPU metrics."""
    metrics = {
        "process_cpu_percent": 0.0,
        "process_memory_rss_mb": 0.0,
        "process_memory_percent": 0.0,
        "system_memory_percent": 0.0,
        "gpu_load_percent": 0.0,
        "gpu_memory_percent": 0.0,
    }

    if psutil is not None:
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent(interval=None)
        mem = process.memory_info()
        virtual = psutil.virtual_memory()

        metrics.update({
            "process_cpu_percent": round(cpu_percent, 1),
            "process_memory_rss_mb": round(mem.rss / 1024 ** 2, 1),
            "process_memory_percent": round(process.memory_percent(), 1),
            "system_memory_percent": round(virtual.percent, 1),
        })

    if GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                metrics["gpu_load_percent"] = round(gpus[0].load * 100, 1)
                metrics["gpu_memory_percent"] = round(gpus[0].memoryUtil * 100, 1)
        except Exception:
            pass

    return metrics


def log_request_metrics(path: str, method: str, latency_s: float, status_code: int) -> None:
    metrics = get_system_metrics()
    log.info(
        f"[Metrics] {method} {path} status={status_code} "
        f"latency_s={latency_s:.3f} cpu={metrics['process_cpu_percent']}% "
        f"rss_mb={metrics['process_memory_rss_mb']} "
        f"mem_pct={metrics['process_memory_percent']} "
        f"sys_mem_pct={metrics['system_memory_percent']} "
        f"gpu={metrics['gpu_load_percent']}% "
        f"gpu_mem={metrics['gpu_memory_percent']}%"
    )

