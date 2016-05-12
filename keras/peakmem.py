#!/usr/bin/env python3

"""Measure peak memory usage for each GPU, according to ``nvidia-smi(1)``."""

from subprocess import check_output
from time import sleep

# Polling interval in seconds. Set to 1/6 because that seems to the fastest
# that nvidia-smi can update (it only goes that speed on some cards, and I have
# no idea whether the K80 or K40 are at 1/6s or longer)
POLLING_INTERVAL = 1/6.0

def get_gpu_usage():
    util_out = check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv']
    )
    util_lines = (l.split()[0] for l in util_out.splitlines()[1:])
    return tuple(map(float, util_lines))

if __name__ == '__main__':
    biggest = None
    while True:
        current = get_gpu_usage()

        if biggest is None:
            biggest = current
            changed = True
        else:
            assert len(current) == len(biggest), \
                "Can't have number of GPUs change :P"
            new_biggest = tuple(max(a, b) for a, b in zip(current, biggest))
            changed = new_biggest != biggest
            biggest = new_biggest

        if changed:
            print(', '.join(map(str, biggest)))

        sleep(POLLING_INTERVAL)
