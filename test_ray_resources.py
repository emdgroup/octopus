"""Test script to understand Ray's behavior when available CPUs < requested CPUs."""

import os
import time

import ray

# Simulate GitHub Actions environment: 2 CPUs available
print("=" * 80)
print("TEST: Ray behavior when 2 CPUs available but 5 tasks each requesting 1 CPU")
print("=" * 80)

# Initialize Ray with only 2 CPUs (like GitHub Actions)
ray.init(num_cpus=2, ignore_reinit_error=True)

print(f"\nRay initialized with {ray.available_resources()}")
print(f"Total CPUs: {ray.available_resources().get('CPU', 0)}")


@ray.remote(num_cpus=1)
def cpu_task(task_id, duration=2):
    """Simulate a CPU-intensive task."""
    print(f"  Task {task_id} started on worker {os.getpid()}")
    time.sleep(duration)
    print(f"  Task {task_id} completed")
    return f"Task {task_id} result"


print("\n" + "=" * 80)
print("Scenario 1: Submit 5 tasks, each requesting 1 CPU")
print("=" * 80)

# Submit 5 tasks
print("\nSubmitting 5 tasks...")
futures = [cpu_task.remote(i, duration=1) for i in range(5)]
print(f"All 5 tasks submitted. Futures created: {len(futures)}")

print("\nWaiting for tasks to complete...")
print("(With 2 CPUs, only 2 tasks can run concurrently)")
print("(Remaining 3 tasks will queue and wait)")

start_time = time.time()
results = ray.get(futures)
elapsed = time.time() - start_time

print(f"\nAll tasks completed in {elapsed:.2f} seconds")
print("Expected time: ~3 seconds (5 tasks / 2 CPUs * 1 sec/task)")
print(f"Results: {results}")

print("\n" + "=" * 80)
print("Scenario 2: What if tasks die/crash?")
print("=" * 80)


@ray.remote(num_cpus=1)
def crashing_task(task_id):
    """Simulate a task that crashes."""
    print(f"  Crashing task {task_id} started")
    if task_id % 2 == 0:
        # Simulate crash
        raise RuntimeError(f"Task {task_id} crashed!")
    time.sleep(0.5)
    return f"Task {task_id} success"


print("\nSubmitting 5 tasks (some will crash)...")
futures = [crashing_task.remote(i) for i in range(5)]

print("\nTrying to get results...")
try:
    results = ray.get(futures)
    print(f"Results: {results}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 80)
print("Scenario 3: Memory pressure simulation")
print("=" * 80)


@ray.remote(num_cpus=1, memory=100 * 1024 * 1024)  # Request 100MB
def memory_task(task_id):
    """Simulate a memory-intensive task."""
    print(f"  Memory task {task_id} started")
    # Allocate some memory
    data = [0] * (10 * 1024 * 1024)  # ~10MB of integers
    time.sleep(0.5)
    return len(data)


print("\nSubmitting 5 memory-intensive tasks...")
futures = [memory_task.remote(i) for i in range(5)]

try:
    results = ray.get(futures, timeout=10)
    print(f"All tasks completed: {len(results)} results")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Cleanup
ray.shutdown()

print("\n" + "=" * 80)
print("Key Findings:")
print("=" * 80)
print("""
1. Ray WILL queue tasks when CPUs are insufficient
   - Only 2 tasks run concurrently with 2 CPUs available
   - Remaining tasks wait in queue

2. If tasks crash/die:
   - Ray propagates the error as ray.exceptions.RayTaskError
   - Or ray.exceptions.WorkerCrashedError if worker dies unexpectedly

3. In CI with limited resources:
   - Workers may crash due to:
     * Memory pressure (OOM)
     * System errors
     * Resource contention
   - This is what's happening in test_analysispipeline.py!
""")
