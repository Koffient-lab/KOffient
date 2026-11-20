#!/usr/bin/env bash
# Speed and memory on the phone: Granite Q4_0 candidate (token table Q6_K) against the same recipe with the token
# table in Q4_0 / Q4_K. 2 interleaved rounds, short prompt, n_ctx 2048, load without mmap, 4 threads.
set -u
export ANDROID_HOME='C:\Android' PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1
ADB=/c/Android/platform-tools/adb.exe
PY=/c/koffient-work/venv/Scripts/python.exe
LOG=/c/koffient-work/results/logs/phone-emb4.log
D='C:\koffient-work\models\granite-4.0-1b\6a7381b'
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
say "start"
for round in 1 2; do
  for r in q4_0-imat-ffn_down-q4_0 q4_0-emb-q4_0-imat-ffn_down-q4_0 q4_0-emb-q4_k-imat-ffn_down-q4_0; do
    say "round $round: $r"
    "$PY" -I -m koffient_tooling bench run --work-dir 'C:\koffient-work' --build android-arm64-dotprod \
      --model "$D\\granite-4.0-1b-$r.gguf" --load-mode none --threads 4 --skip-cold --timeout 1800 --n-ctx 2048 \
      --label "phone-emb4-r$round" >> "$LOG" 2>&1
    say "done (exit $?)"
    "$ADB" shell "rm -rf /data/local/tmp/koffient/runner/models" < /dev/null
    sleep 60
  done
done
say "all done"
