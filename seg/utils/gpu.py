import subprocess

def get_gpu_memroy(gpu_id:set = None):

    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], capture_output=True)
    output = result.stdout.decode("utf-8").strip().split("\n")
    memeory_info = []
    for i, line in enumerate(output):
        if gpu_id is not None and i not in gpu_id:
            continue
        memory_used, memory_total = map(lambda x: int(x.split(" ")[0]), line.split(", "))
        info = dict(gpu_id=i, memory_used=memory_used, memory_total=memory_total)
        memeory_info.append(info)
    return memeory_info