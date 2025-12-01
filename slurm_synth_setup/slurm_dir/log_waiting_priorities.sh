#!/bin/bash
OUT=waiting_priorities.log

echo "=== Starting priority log: $(date) ===" >> $OUT

while true; do
  echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> $OUT

  # 1) Log total priority for each pending job
  squeue -t PD -h -o "JobID=%i Priority=%Q JobName=%j" >> $OUT

  # 2) (Optional) Log factor breakdown for those same jobs
  JOB_IDS=$(squeue -t PD -h -o "%i" | tr '\n' ',' | sed 's/,$//')
  if [ -n "$JOB_IDS" ]; then
    echo "--- Priority factor breakdown ---" >> $OUT
    sprio -j "$JOB_IDS" >> $OUT
  fi

  echo "" >> $OUT
  sleep 60
done
