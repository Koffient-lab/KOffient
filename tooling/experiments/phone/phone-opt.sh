#!/usr/bin/env bash
# Phone optimisation experiments, run after phone-check.sh finishes (never at the same time: they would distort
# each other). One model file on the phone at a time (the phone has ~1.9 GB free). Order G, Q, G0, G again: the
# second Granite pass shows drift (heat, battery) over the session.
set -u
export MSYS_NO_PATHCONV=1
ADB=/c/Android/platform-tools/adb.exe
LOG=/c/koffient-work/results/logs/phone-opt.log
CHECK_LOG=/c/koffient-work/results/logs/phone-check.log
R=/data/local/tmp/koffient/opt
OUT="C:\\koffient-work\\results\\raw\\phone-opt\\$(date -u +%Y%m%dT%H%M%SZ)"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

# Start this only after phone-check.sh has exited (it is launched by hand on its completion notice). A first
# version waited for "all done" in phone-check.log, matched the line a DRY run had written, and overlapped the series.
pgrep_check=$(powershell -NoProfile -c "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'phone-check.sh' -and \$_.Name -eq 'bash.exe' }).Count")
[ "${pgrep_check//[^0-9]/}" = "0" ] || { echo "phone-check.sh is still running"; exit 1; }
say "start"
# The series run phone-tools-r1 of Granite overlapped the first, wrong start of this script: measure it again.
say "re-run of Granite phone-tools (r1 overlapped): phone-tools-r4"
ANDROID_HOME='C:\Android' PYTHONIOENCODING=utf-8 /c/koffient-work/venv/Scripts/python.exe -I -m koffient_tooling bench run \
  --work-dir 'C:\koffient-work' --build android-arm64-dotprod \
  --model 'C:\koffient-work\release\granite-4.0-1b-q4_k_m\granite-4.0-1b-q4_k_m.gguf' --load-mode none --threads 4 \
  --skip-cold --timeout 1800 --label phone-tools-r4 --n-ctx 2048 --prompt-file ttft_tools_en.txt >> "$LOG" 2>&1
"$ADB" shell "rm -rf /data/local/tmp/koffient/runner/models"
sleep 90
"$ADB" shell "mkdir -p $R/out && rm -f $R/out/*"
"$ADB" push 'C:\koffient-work\opt\opt_device.sh' "$R/opt_device.sh" >/dev/null
"$ADB" push 'C:\koffient-work\opt\tools_q1.txt' "$R/tools_q1.txt" >/dev/null
"$ADB" push 'C:\koffient-work\opt\tools_q2.txt' "$R/tools_q2.txt" >/dev/null
"$ADB" shell "chmod 755 $R/opt_device.sh"

one() {  # local model, tag, knobs
  local m=$1 tag=$2 knobs=$3 remote="$R/model.gguf"
  say "$tag: push $(basename "$m")"
  "$ADB" push "$m" "$remote" >/dev/null || { say "$tag: push failed"; return; }
  say "$tag: run (battery $("$ADB" shell dumpsys battery | tr -d '\r' | awk '/level/ {l=$2} /temperature/ {t=$2/10} END {print l "%, " t " C"}'))"
  "$ADB" shell "$R/opt_device.sh $remote $tag $knobs"
  "$ADB" shell "cat $R/out/$tag.done 2>/dev/null" | grep -q done && say "$tag: done" || say "$tag: NOT done"
  "$ADB" shell "rm -f $remote"
  sleep 60
}
one 'C:\koffient-work\release\granite-4.0-1b-q4_k_m\granite-4.0-1b-q4_k_m.gguf' g-q4km 1
one 'C:\koffient-work\release\qwen3-0.6b-q4_k_m-tied\qwen3-0.6b-tied-q4_k_m-tied.gguf' q-q4km 0
one 'C:\koffient-work\models\granite-4.0-1b\6a7381b\granite-4.0-1b-q4_0-imat-ffn_down-q4_0.gguf' g-q40 1
one 'C:\koffient-work\release\granite-4.0-1b-q4_k_m\granite-4.0-1b-q4_k_m.gguf' g-q4km-2 0

# Memory: Granite variants against the series run phone-ctx4096 (tool prompt, n_ctx 4096, f16 KV, ubatch 512).
mem() {  # model label extra-args...
  local m=$1 label=$2; shift 2
  say "memory run $label"
  ANDROID_HOME='C:\Android' PYTHONIOENCODING=utf-8 /c/koffient-work/venv/Scripts/python.exe -I -m koffient_tooling \
    bench run --work-dir 'C:\koffient-work' --build android-arm64-dotprod --model "$m" --load-mode none --threads 4 \
    --skip-cold --timeout 1800 --n-ctx 4096 --prompt-file ttft_tools_en.txt --label "$label" "$@" >> "$LOG" 2>&1
  say "memory run $label done (exit $?)"
  sleep 90
}
GR='C:\koffient-work\release\granite-4.0-1b-q4_k_m\granite-4.0-1b-q4_k_m.gguf'
mem "$GR" phone-mem-kvq8 --kv-cache-type q8_0
mem "$GR" phone-mem-ub128 --ubatch 128
mem "$GR" phone-mem-kvq8-ub128 --kv-cache-type q8_0 --ubatch 128
"$ADB" shell "rm -rf /data/local/tmp/koffient/runner/models"
mem 'C:\koffient-work\release\qwen3-0.6b-q4_k_m-tied\qwen3-0.6b-tied-q4_k_m-tied.gguf' phone-mem-kvq8 --kv-cache-type q8_0
"$ADB" shell "rm -rf /data/local/tmp/koffient/runner/models"
mem 'C:\koffient-work\models\granite-4.0-1b\6a7381b\granite-4.0-1b-q4_k_m-emb-q4_k-imat.gguf' phone-mem-emb-q4k
"$ADB" shell "rm -rf /data/local/tmp/koffient/runner/models"

mkdir -p "$(cygpath -u "$OUT")"
"$ADB" pull "$R/out" "$OUT" >/dev/null && say "pulled to $OUT"
"$ADB" shell "rm -rf $R"
say "storage after cleanup: $("$ADB" shell df -h /data | tr -d '\r' | tail -1)"
say "all done"
