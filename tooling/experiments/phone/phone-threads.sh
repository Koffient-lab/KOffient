#!/usr/bin/env bash
# Do the little cores help? On the 9R both prompt and generation scale linearly over 1-4 big-core threads, so they
# are compute-bound. Granite q4_k_m, 4/5/6/8 threads with no core mask. Starts after phone-cache.sh.
set -u
export MSYS_NO_PATHCONV=1
ADB=/c/Android/platform-tools/adb.exe
LOG=/c/koffient-work/results/logs/phone-threads.log
CACHE_LOG=/c/koffient-work/results/logs/phone-cache.log
OUT=/c/koffient-work/results/raw/phone-threads
R=/data/local/tmp/koffient/threads
BIN=/data/local/tmp/koffient/runner/bin/android-arm64-dotprod
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
mkdir -p "$OUT"
n0=$(grep -c "all done" "$CACHE_LOG" 2>/dev/null || echo 0)
until [ "$(grep -c "all done" "$CACHE_LOG" 2>/dev/null || echo 0)" -gt "$n0" ]; do sleep 30; done
say "start"
"$ADB" shell "mkdir -p $R"
"$ADB" push 'C:\koffient-work\release\granite-4.0-1b-q4_k_m\granite-4.0-1b-q4_k_m.gguf' "$R/model.gguf" >/dev/null
"$ADB" shell "cd $R && LD_LIBRARY_PATH=$BIN $BIN/llama-bench -m $R/model.gguf -t 4,5,6,8 -p 512 -n 64 -r 3 -lm none -o json \
  > $R/threads.json 2> $R/threads.err"
"$ADB" pull "$R/threads.json" "$(cygpath -w "$OUT/threads.json")" >/dev/null
"$ADB" pull "$R/threads.err" "$(cygpath -w "$OUT/threads.err")" >/dev/null
"$ADB" shell "rm -rf $R"
say "all done"
