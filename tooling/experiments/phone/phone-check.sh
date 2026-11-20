#!/usr/bin/env bash
# Phone numbers for the two MVP packages on a real Android phone (USB debugging on, one device connected).
#   bash phone-check.sh            check the phone, then measure (about 1-1.5 h)
#   DRY=1 bash phone-check.sh      check the phone and print the commands only
# Same protocol as the emulator series: 3 interleaved rounds, short prompt and the six-tool request,
# loading without mmap, n_ctx 2048; then one run per model at n_ctx 4096 for memory. Cold starts need root: skipped.
# The phone should be charged, on the desk (not in a case), nothing else running. The script keeps the screen on by
# sending a wake-up key every 15 s (OnePlus does not let adb change the screen timeout) and removes the model files
# from the phone at the end.
set -u
export ANDROID_HOME='C:\Android' PYTHONIOENCODING=utf-8
ADB=/c/Android/platform-tools/adb.exe
PY=/c/koffient-work/venv/Scripts/python.exe
LOG=/c/koffient-work/results/logs/phone-check.log
BUILD=android-arm64-dotprod          # armv8.2-a+dotprod+fp16, no i8mm
G='C:\koffient-work\release\granite-4.0-1b-q4_k_m\granite-4.0-1b-q4_k_m.gguf'
Q='C:\koffient-work\release\qwen3-0.6b-q4_k_m-tied\qwen3-0.6b-tied-q4_k_m-tied.gguf'
REMOTE_MODELS=/data/local/tmp/koffient/runner/models
COOL=90                              # seconds between runs, so throttling from one run does not hit the next
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
sh_() { MSYS_NO_PATHCONV=1 "$ADB" shell "$@" | tr -d '\r'; }

n=$("$ADB" devices | tr -d '\r' | awk 'NR > 1 && $2 == "device"' | wc -l)
[ "$n" -eq 1 ] || { echo "need exactly one device in 'adb devices', found $n"; exit 1; }

say "phone: $(sh_ getprop ro.product.manufacturer) $(sh_ getprop ro.product.model), Android $(sh_ getprop ro.build.version.release), SoC $(sh_ getprop ro.soc.model)"
say "RAM: $(sh_ cat /proc/meminfo | awk '/MemTotal|MemAvailable/ {printf "%s %d MB  ", $1, $2/1024}')"
say "storage: $(sh_ df -h /data | tail -1)"
features=$(sh_ cat /proc/cpuinfo | awk -F': ' '/^Features/ {print $2; exit}')
say "CPU features: $features"
case " $features " in *" asimddp "*) ;; *) say "STOP: no dotprod (asimddp); the $BUILD build would crash"; exit 1 ;; esac
case " $features " in *" i8mm "*) say "note: i8mm present; the i8mm build could be faster here (not used, the SDK ships $BUILD)" ;; esac

# Big cores: max frequency within 70% of the fastest core (Snapdragon 870: 1 x 3.19 + 3 x 2.42 GHz -> 4 threads).
freqs=$(sh_ 'for f in /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq; do cat $f; done')
top=$(echo "$freqs" | sort -n | tail -1)
THREADS=$(echo "$freqs" | awk -v top="$top" '$1 >= 0.7 * top {n++} END {print n}')
say "core max freqs (kHz): $(echo $freqs) -> threads $THREADS"

if [ "${DRY:-0}" != 1 ]; then
  ( while true; do MSYS_NO_PATHCONV=1 "$ADB" shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1; sleep 15; done ) &
  WAKER=$!
  trap 'kill $WAKER 2>/dev/null' EXIT
fi

wait_device() {  # the USB link dropped once; wait up to 10 min for the phone to come back
  local i
  for i in $(seq 60); do
    "$ADB" devices | tr -d '\r' | awk 'NR > 1 && $2 == "device"' | grep -q . && return 0
    [ "$i" = 1 ] && say "phone not connected, waiting"
    sleep 10
  done
  say "STOP: phone did not come back"; exit 1
}

run() {  # model label args...
  local m=$1 label=$2; shift 2
  local cmd=("$PY" -I -m koffient_tooling bench run --work-dir 'C:\koffient-work' --build "$BUILD" --model "$m"
             --load-mode none --threads "$THREADS" --skip-cold --timeout 1800 --label "$label" "$@")
  if [ "${DRY:-0}" = 1 ]; then echo "${cmd[*]}"; return; fi
  wait_device
  say "bench $label $(basename "$m")  battery $(sh_ dumpsys battery | awk '/level/ {l=$2} /temperature/ {t=$2/10} END {print l "%, " t " C"}')"
  "${cmd[@]}" >> "$LOG" 2>&1
  say "done $label (exit $?)"
  sleep "$COOL"
}
for round in 1 2 3; do
  run "$G" "phone-r$round" --n-ctx 2048
  run "$Q" "phone-r$round" --n-ctx 2048
  run "$G" "phone-tools-r$round" --n-ctx 2048 --prompt-file ttft_tools_en.txt
  run "$Q" "phone-tools-r$round" --n-ctx 2048 --prompt-file ttft_tools_en.txt
done
run "$G" "phone-ctx4096" --n-ctx 4096 --prompt-file ttft_tools_en.txt
run "$Q" "phone-ctx4096" --n-ctx 4096 --prompt-file ttft_tools_en.txt
if [ "${DRY:-0}" != 1 ]; then
  wait_device
  sh_ "rm -rf $REMOTE_MODELS" && say "model files removed from the phone; storage: $(sh_ df -h /data | tail -1)"
fi
say "all done"
