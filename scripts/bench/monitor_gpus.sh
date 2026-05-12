#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Background GPU monitor: polls `nvidia-smi` every $INTERVAL_S seconds and
# appends one row per GPU per sample to $OUT_CSV.
#
# Columns: epoch_s, iso_ts, index, util_gpu_pct, util_mem_pct,
#          mem_used_mib, mem_total_mib, power_w, temp_c, sm_clock_mhz
#
# Usage:
#   bash monitor_gpus.sh OUT_CSV [INTERVAL_S]
#
# Designed to be run in the background. The polled `nvidia-smi` CSV format is
# stable across driver versions, so the resulting file can be parsed by the
# `summarize.py` companion script regardless of host driver.
# ---------------------------------------------------------------------------
set -euo pipefail

OUT_CSV="${1:?usage: monitor_gpus.sh OUT_CSV [INTERVAL_S]}"
INTERVAL_S="${2:-1}"

mkdir -p "$(dirname "${OUT_CSV}")"

# Header (only if the file is empty / new).
if [[ ! -s "${OUT_CSV}" ]]; then
  echo "epoch_s,iso_ts,index,util_gpu_pct,util_mem_pct,mem_used_mib,mem_total_mib,power_w,temp_c,sm_clock_mhz" > "${OUT_CSV}"
fi

# Trap so a `kill <pid>` flushes any in-flight buffered writes.
trap 'exit 0' INT TERM

# Loop. Each `nvidia-smi --query-gpu=...` returns one line per GPU; we prefix
# with the epoch and iso timestamp captured before the call.
while true; do
  EPOCH_S="$(date +%s)"
  ISO_TS="$(date -u +%FT%TZ)"
  # `--format=csv,noheader,nounits` strips header and unit suffixes for clean numerics.
  if ! nvidia-smi \
        --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
        --format=csv,noheader,nounits 2>/dev/null \
       | awk -v t="${EPOCH_S}" -v iso="${ISO_TS}" -F',' 'BEGIN{OFS=","} {
            # `nvidia-smi --format=csv,noheader,nounits` separates fields with
            # ", " — split on commas and strip the leading space from each field.
            for (i=1;i<=NF;i++) gsub(/^[ \t]+|[ \t]+$/,"",$i);
            print t,iso,$1,$2,$3,$4,$5,$6,$7,$8
         }' >> "${OUT_CSV}"; then
    # If nvidia-smi failed transiently (driver flap, etc.) record an error row.
    echo "${EPOCH_S},${ISO_TS},NA,NA,NA,NA,NA,NA,NA,NA" >> "${OUT_CSV}"
  fi
  sleep "${INTERVAL_S}"
done
