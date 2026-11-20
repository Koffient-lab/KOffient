#!/usr/bin/env bash
# Where does the time go on the phone? simpleperf (on the phone, perf_event_paranoid -1) over llama-bench, 4 threads,
# prompt-only (pp512) and generation-only (tg128) separately, Granite q4_k_m and q4_0-imat. Reports per thread, per
# CPU and per function (ggml_barrier share = threads waiting for each other). The binary keeps its symbols.
# Starts after phone-threads.sh writes "all done".
set -u
export MSYS_NO_PATHCONV=1
ADB=/c/Android/platform-tools/adb.exe
LOG=/c/koffient-work/results/logs/phone-profile.log
THREADS_LOG=/c/koffient-work/results/logs/phone-threads.log
OUT=/c/koffient-work/results/raw/phone-profile
R=/data/local/tmp/koffient/profile
BIN=/data/local/tmp/koffient/runner/bin/android-arm64-dotprod
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
mkdir -p "$OUT"
n0=$(grep -c "all done" "$THREADS_LOG" 2>/dev/null || echo 0)
until [ "$(grep -c "all done" "$THREADS_LOG" 2>/dev/null || echo 0)" -gt "$n0" ]; do sleep 30; done
say "start"
"$ADB" shell "mkdir -p $R"

one() {  # local model, tag
  local m=$1 tag=$2
  say "$tag: push"
  "$ADB" push "$m" "$R/model.gguf" >/dev/null
  for mode in pp tg; do
    if [ $mode = pp ]; then args="-p 512 -n 0"; else args="-p 0 -n 128"; fi
    say "$tag $mode: record"
    "$ADB" shell "cd $R && simpleperf record -e cpu-clock -f 4000 -o $R/$tag-$mode.data \
      $BIN/llama-bench -m $R/model.gguf -t 4 $args -r 2 -lm none -o json > $R/$tag-$mode.json 2> $R/$tag-$mode.err" \
      < /dev/null >> "$LOG" 2>&1
    for sort in "tid,comm" "cpu" "symbol"; do
      name=$(echo "$sort" | tr ',' '_')
      "$ADB" shell "cd $R && simpleperf report -i $R/$tag-$mode.data --sort $sort --percent-limit 0.5 \
        > $R/$tag-$mode-$name.txt 2>&1" < /dev/null
      "$ADB" pull "$R/$tag-$mode-$name.txt" "$(cygpath -w "$OUT/$tag-$mode-$name.txt")" >/dev/null
    done
    "$ADB" pull "$R/$tag-$mode.json" "$(cygpath -w "$OUT/$tag-$mode.json")" >/dev/null
    "$ADB" shell "rm -f $R/$tag-$mode.data"
    sleep 60
  done
  "$ADB" shell "rm -f $R/model.gguf"
}
one 'C:\koffient-work\release\granite-4.0-1b-q4_k_m\granite-4.0-1b-q4_k_m.gguf' g-q4km
one 'C:\koffient-work\models\granite-4.0-1b\6a7381b\granite-4.0-1b-q4_0-imat-ffn_down-q4_0.gguf' g-q40
"$ADB" shell "rm -rf $R"
say "all done"
