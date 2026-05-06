# GPU Usage Logging

The system log now includes GPU metrics monitoring alongside the existing CPU and Memory metrics. This allows observing processing bottlenecks and memory consumptions specific to GPU workload in operations like RAG or model inference.

## How it works

The changes were made in `apps/modeling/metrics.py`. We have integrated standard GPU monitoring using the `GPUtil` Python library.

1.  **Dependencies (`GPUtil`)**:
    The system attempts to import the `GPUtil` library. By wrapping the import in a `try...except` block, the system maintains reliability. If `GPUtil` is not installed or no GPU is available, the system will fallback safely to `0.0%` usage without crashing. To actually enable extraction, make sure `GPUtil` is installed in your python environment:
    ```bash
    pip install GPUtil
    ```

2.  **`get_system_metrics()` Function**:
    This function was updated to initialize default GPU metrics: `gpu_load_percent` and `gpu_memory_percent`.
    If `GPUtil` is successfully imported, it retrieves the GPU device array. If at least one compatible NVIDIA GPU is found on the host machine (`gpus[0]`), it resolves the load (utilization percentage) and video memory use percentage and updates the dictionary accordingly. All outputs are rounded to 1 decimal place.

3.  **`log_request_metrics()` Function**:
    This function has been expanded. Previously, it logged basic process CPU use, process RSS memory, and system memory. Now it appends `gpu=...% gpu_mem=...%` to the log string using variables safely extracted via `get_system_metrics()`.

## Output Format Example

When a request metric is logged via the `alia.metrics` logger, the output will look something like this:

```text
[Metrics] GET /api/v1/some-endpoint status=200 latency_s=0.420 cpu=12.5% rss_mb=85.2 mem_pct=0.5 sys_mem_pct=45.0 gpu=84.5% gpu_mem=27.6%
```

If a compatible GPU is not present or `GPUtil` isn't installed, the GPU metrics will transparently evaluate and appear as `gpu=0.0% gpu_mem=0.0%`.
