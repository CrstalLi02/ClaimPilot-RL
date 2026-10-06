import torch
import os
import sys
import time
import argparse
import subprocess
import threading


def get_visible_gpu_indices():
    """
    Return the list of physical GPU IDs actually visible to the current process.
    The cluster isolates GPUs via CUDA_VISIBLE_DEVICES; this is required to query the matching physical GPU.
    If CUDA_VISIBLE_DEVICES is not set, return [0, 1, ..., gpu_number-1].
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and cvd != "NoDevFiles":
        return [int(x.strip()) for x in cvd.split(",") if x.strip().isdigit()]
    # Probe with torch when not set
    n = torch.cuda.device_count()
    return list(range(n))


def gpu_utilization(physical_gpu_id):
    """Query the current utilization (%) of the given physical GPU via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits",
             f"--id={physical_gpu_id}"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out)
    except Exception:
        return 0


def keep_gpu_busy(logical_i, phys_i, threshold, stop_event):
    """
    One independent thread per GPU, continuously monitoring and compensating.
    - When utilization < threshold, start matmul compensation immediately
    - When utilization >= threshold, sleep briefly and check again
    - Check frequently during compensation (once every 10 matmuls) to avoid overcompensating
    """
    device = torch.device(f"cuda:{logical_i}")
    # A100 80GB: an 8192×8192 float16 matmul effectively raises utilization, using ~0.5GB of memory
    sz = 8192
    a = torch.rand(sz, sz, dtype=torch.float16, device=device)
    b = torch.rand(sz, sz, dtype=torch.float16, device=device)

    print(f"[low_gpu_util] thread started: logical={logical_i} physical={phys_i} threshold={threshold}%", flush=True)

    while not stop_event.is_set():
        util = gpu_utilization(phys_i)
        if util < threshold:
            # Utilization too low, keep running matmul compensation
            count = 0
            while not stop_event.is_set():
                # Run 10 matmuls in a row before rechecking utilization (reduces nvidia-smi overhead)
                for _ in range(10):
                    a = a @ b
                count += 1
                if count % 3 == 0:
                    torch.cuda.synchronize(logical_i)
                    cur_util = gpu_utilization(phys_i)
                    if cur_util >= threshold:
                        print(f"[low_gpu_util] GPU{phys_i} utilization recovered: {cur_util}% >= {threshold}%", flush=True)
                        break
        else:
            # Utilization is on target, sleep briefly (0.5s) before checking again
            # Note: don't sleep too long, or 2s+ idle periods during training will be missed
            time.sleep(0.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_number", required=False, default=1, type=int)
    parser.add_argument("--gpu_utilization_threshold", required=False, default=70, type=int)
    args = parser.parse_args()

    # Get the actual physical GPU IDs (handles cluster CUDA_VISIBLE_DEVICES isolation)
    physical_ids = get_visible_gpu_indices()
    # Take only the first gpu_number
    physical_ids = physical_ids[:args.gpu_number]
    print(f"[low_gpu_util] monitoring physical GPUs: {physical_ids}, threshold={args.gpu_utilization_threshold}%", flush=True)

    stop_event = threading.Event()

    # ★ One independent thread per GPU, monitoring and compensating in parallel (the original serial version left some GPUs persistently underutilized)
    threads = []
    for logical_i, phys_i in enumerate(physical_ids):
        t = threading.Thread(
            target=keep_gpu_busy,
            args=(logical_i, phys_i, args.gpu_utilization_threshold, stop_event),
            daemon=True,
            name=f"gpu-keeper-{phys_i}",
        )
        t.start()
        threads.append(t)

    try:
        # Main thread waits forever; daemon threads exit automatically when killed
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[low_gpu_util] interrupt received, stopping all daemon threads", flush=True)
        stop_event.set()
        for t in threads:
            t.join(timeout=3)
