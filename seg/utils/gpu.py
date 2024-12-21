import subprocess


def get_gpu_memroy():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                            capture_output=True)
    output = result.stdout.decode("utf-8").strip().split("\n")
    info = ''
    for i, line in enumerate(output):
        memory_used, memory_total = map(lambda x: int(x.split(" ")[0]), line.split(", "))
        info += f"GPU-{i}: {memory_used/1024:.2f} / {memory_total//1024} GB   "
    return info
