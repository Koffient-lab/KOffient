#!/usr/bin/env bash
# Is the clock cap on the big cores per process or a power budget? Runs in the 90 s cool-down after the next
# finished phone-check run: idle clocks, then 1 and 4 busy threads left to the scheduler (pinning fails: the kernel
# takes idle big cores offline), 8 s each, sampling clocks and online cores. ~25 s, so the next series run still
# gets ~60 s of rest.
set -u
export MSYS_NO_PATHCONV=1
ADB=/c/Android/platform-tools/adb.exe
CHECK_LOG=/c/koffient-work/results/logs/phone-check.log
OUT=/c/koffient-work/results/logs/burn-test.log
n0=$(grep -c " done " "$CHECK_LOG")
until [ "$(grep -c " done " "$CHECK_LOG")" -gt "$n0" ]; do sleep 2; done
echo "$(date '+%Y-%m-%d %H:%M:%S') round 2, after: $(grep " done " "$CHECK_LOG" | tail -1)" >> "$OUT"
"$ADB" shell '
f() { echo "$1: online $(cat /sys/devices/system/cpu/online), little $(( $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq) / 1000 )), mid $(( $(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq) / 1000 )), prime $(( $(cat /sys/devices/system/cpu/cpufreq/policy7/scaling_cur_freq) / 1000 )) MHz; busy on: $(ps -A -o PSR,NAME | grep -w yes | cut -c1-4 | tr -d " " | tr "\n" ",")"; }
f idle
yes > /dev/null & a=$!
sleep 4; f "1 thread, 4 s"; sleep 4; f "1 thread, 8 s"
kill $a; sleep 1
for i in 1 2 3 4; do yes > /dev/null & done
sleep 4; f "4 threads, 4 s"; sleep 4; f "4 threads, 8 s"
pkill -x yes; sleep 1; f "after"
' >> "$OUT" 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') end" >> "$OUT"
