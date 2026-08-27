#!/bin/sh

# Fixed-function root wrapper for Apple hardware-counter capture. This file is
# copied to /usr/local/sbin as a root-owned executable by the companion
# installer; do not grant passwordless sudo to the repository copy.

set -eu

if [ "$#" -gt 1 ]; then
    /usr/bin/printf '%s\n' \
        "usage: pyfluid-powermetrics [15|30|60|120|300]" >&2
    exit 64
fi

duration_seconds=${1:-60}
case "$duration_seconds" in
    15|30|60|120|300) ;;
    *)
        /usr/bin/printf '%s\n' \
            "duration must be one of 15, 30, 60, 120, or 300 seconds" >&2
        exit 64
        ;;
esac

if [ "$(/usr/bin/uname -s)" != "Darwin" ]; then
    /usr/bin/printf '%s\n' "pyfluid-powermetrics requires macOS" >&2
    exit 69
fi

sample_rate_ms=500
sample_count=$((duration_seconds * 1000 / sample_rate_ms))

exec /usr/bin/powermetrics \
    --samplers tasks,cpu_power \
    --show-process-amp \
    --show-process-ipc \
    --show-cpu-qos \
    --sample-rate "$sample_rate_ms" \
    --sample-count "$sample_count"
