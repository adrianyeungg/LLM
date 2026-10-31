import time
import pandas as pd
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates

from PythonTesting.gpt.sisyphus import prompt

try:
    nvmlInit()
    gpu_available = True
    device_handle = nvmlDeviceGetHandleByIndex(0)
except Exception:
    gpu_available = False
    print("no cuda, switching to cpu")

performance_logs = []


def track_llm_inference(prompt_text, tokens_in, expected_tokens_out):
    start_time = time.time()
    start_mem = nvmlDeviceGetMemoryInfo(device_handle).used if gpu_available else 0

    import sisyphus
    response = sisyphus
    time.sleep(0.4)  # Simulating processing time

    end_time = time.time()
    end_mem = nvmlDeviceGetMemoryInfo(device_handle).used if gpu_available else 0
    gpu_util = nvmlDeviceGetUtilizationRates(device_handle).gpu if gpu_available else 0

    duration = end_time - start_time
    tokens_per_sec = expected_tokens_out / duration if duration > 0 else 0
    mem_used_gb = end_mem / (1024 ** 3)

    performance_logs.append({
        "input_tokens": tokens_in,
        "latency_sec": duration,
        "throughput_tokens_per_sec": tokens_per_sec,
        "gpu_utilization_pct": gpu_util,
        "gpu_memory_used_gb": mem_used_gb
    })


sample_runs = [(128, 50), (512, 150), (1024, 250)]
for tokens_in, tokens_out in sample_runs:
    track_llm_inference("Dummy prompt text...", tokens_in, tokens_out)


df = pd.DataFrame(performance_logs)
df.to_csv("llm_performance_metrics.csv", index=False)
print("saved")

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("llm_performance_metrics.csv")

fig, ax1 = plt.subplots(figsize=(10, 5))

color = '#1f77b4'
ax1.set_xlabel('Input Tokens')
ax1.set_ylabel('Throughput (Tokens/sec)', color=color)
ax1.plot(df['input_tokens'], df['throughput_tokens_per_sec'], marker='o', color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = '#ff7f0e'
ax2.set_ylabel('GPU Memory Used (GB)', color=color)
ax2.plot(df['input_tokens'], df['gpu_memory_used_gb'], marker='s', color=color, linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Custom LLM Local Performance Evaluation')
fig.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()
