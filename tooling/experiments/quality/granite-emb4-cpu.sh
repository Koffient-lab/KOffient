#!/usr/bin/env bash
# Stage 1 of the 4-bit token table check on the CPU (the GPU was taken; ~2.6 GB RAM free, so one server):
# candidate q4_0-imat-ffn_down-q4_0 and the same recipe with token_embd Q4_0, on the main and short tasks, task by
# task in turn, so a pairwise comparison on one engine is possible.
set -u
W='C:\koffient-work'; PY=/c/koffient-work/venv/Scripts/python.exe; LOG=/c/koffient-work/results/logs
NLOG="$LOG/granite-emb4-cpu.log"
D='C:\koffient-work\models\granite-4.0-1b\6a7381b'
export PYTHONIOENCODING=utf-8
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$NLOG"; }
say "start"
for t in bfcl-en bfcl-ru jsonschema mmlu-en mmlu-ru; do
  for r in q4_0-imat-ffn_down-q4_0 q4_0-emb-q4_0-imat-ffn_down-q4_0; do
    say "$t $r (free RAM $(powershell -NoProfile -c "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" | tr -d '\r') GB)"
    "$PY" -I -m koffient_tooling quality run --work-dir "$W" --model "$D\\granite-4.0-1b-$r.gguf" --tasks "$t" \
      --servers 1 --threads 8 >> "$LOG/granite-emb4-cpu-fast.log" 2>&1
    say "done (exit $?)"
  done
done
say "all done"
