#!/system/bin/sh
# Runs on the phone. usage: opt_device.sh <model on device> <tag> <knobs 0|1>
# 1) scaling: prompt (512) and generation (64) at 1-4 threads, pinned to the big cores (cpu4-7);
# 2) knobs (optional): micro-batch 128/256/512 x flash attention on/off at 4 threads;
# 3) prompt cache: build the state of the tool prompt once, then answer a different question with the same prefix
#    from the cache (3x) and without it (3x).
BIN=/data/local/tmp/koffient/runner/bin/android-arm64-dotprod
P=/data/local/tmp/koffient/opt
O=$P/out
mkdir -p $O
M=$1; T=$2; K=$3
TS="taskset f0"
$TS true 2>/dev/null || TS=""
export LD_LIBRARY_PATH=$BIN
echo "taskset: ${TS:-none}" > $O/$T.info
state() {  # clocks of the three core clusters and the thermal status (0 none ... 4 critical)
  echo "$(date +%H:%M:%S) thermal $(dumpsys thermalservice 2>/dev/null | grep -m1 'Thermal Status' | tr -dc 0-9)" \
       "MHz $(for p in 0 4 7; do echo -n "$(( $(cat /sys/devices/system/cpu/cpufreq/policy$p/scaling_cur_freq) / 1000 )) "; done)"
}

state > $O/$T-scale.state
$TS $BIN/llama-bench -m $M -t 1,2,3,4 -p 512 -n 64 -r 3 -lm none -o json > $O/$T-scale.json 2> $O/$T-scale.err
state >> $O/$T-scale.state
sleep 60
if [ "$K" = 1 ]; then
  $TS $BIN/llama-bench -m $M -t 4 -p 512 -n 0 -ub 128,256,512 -fa on,off -r 3 -lm none -o json \
    > $O/$T-knobs.json 2> $O/$T-knobs.err
  sleep 60
  # Thread pool: spin-waiting (poll) keeps every core at 98% and heats the phone, which throttles to ~73% of the
  # top clock; strict pinning; thread priority.
  state > $O/$T-threads.state
  $TS $BIN/llama-bench -m $M -t 4 -p 512 -n 64 -r 3 -lm none --poll 0,50,100 -o json \
    > $O/$T-poll.json 2> $O/$T-poll.err
  sleep 60
  $BIN/llama-bench -m $M -t 4 -p 512 -n 64 -r 3 -lm none -C f0 --cpu-strict 0,1 -o json \
    > $O/$T-strict.json 2> $O/$T-strict.err
  sleep 60
  $TS $BIN/llama-bench -m $M -t 4 -p 512 -n 64 -r 3 -lm none --prio 0,1,2 -o json \
    > $O/$T-prio.json 2> $O/$T-prio.err
  state >> $O/$T-threads.state
  sleep 60
fi

C=$P/$T.cache
rm -f $C
comp() {  # name prompt [extra flags]
  n=$1; p=$2; shift 2
  s=$(date +%s%N)
  $TS $BIN/llama-completion -m $M -t 4 -c 2048 -n 1 --temp 0 -s 1 -no-cnv --no-display-prompt -lm none -f $p "$@" \
    > $O/$T-$n.out 2> $O/$T-$n.err < /dev/null
  rc=$?
  e=$(date +%s%N)
  echo "$n rc $rc wall_ms $(( (e - s) / 1000000 ))" >> $O/$T-cache.txt
}
comp build $P/tools_q1.txt --prompt-cache $C
ls -l $C >> $O/$T-cache.txt
for i in 1 2 3; do comp cached$i $P/tools_q2.txt --prompt-cache $C --prompt-cache-ro; done
for i in 1 2 3; do comp nocache$i $P/tools_q2.txt; done
rm -f $C
echo done > $O/$T.done
