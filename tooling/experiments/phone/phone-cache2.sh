#!/usr/bin/env bash
# Prompt cache with the model resident in memory (llama-server on the phone) - second run. The first
# run (phone-cache.sh) stopped servers with `pkill -f <full path>`, which never matched `./llama-server`: the Granite
# f16 server kept the port and answered every later probe, so only granite-f16 there is valid. Here: stop by process
# name, check the port is free before each start, and log which model and KV type the server really loaded.
# Starts after phone-profile.sh writes "all done".
set -u
export MSYS_NO_PATHCONV=1
ADB=/c/Android/platform-tools/adb.exe
PY=/c/koffient-work/venv/Scripts/python.exe
LOG=/c/koffient-work/results/logs/phone-cache2.log
WAIT_LOG=/c/koffient-work/results/logs/phone-profile.log
OUT=/c/koffient-work/results/raw/phone-cache
BIN=/data/local/tmp/koffient/cache
PORT=8899
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
mkdir -p "$OUT"
n0=$(grep -c "all done" "$WAIT_LOG" 2>/dev/null || echo 0)
until [ "$(grep -c "all done" "$WAIT_LOG" 2>/dev/null || echo 0)" -gt "$n0" ]; do sleep 30; done
say "start"
"$ADB" shell "pkill -x llama-server; mkdir -p $BIN/slots" < /dev/null
"$ADB" push 'C:\koffient-work\build\android-arm64-dotprod\bin\llama-server' "$BIN/llama-server" >/dev/null
"$ADB" shell "chmod 755 $BIN/llama-server"
"$ADB" forward tcp:$PORT tcp:$PORT

probe() {  # tag, kv type
  local tag=$1 kv=$2
  "$ADB" shell "pkill -x llama-server; sleep 2" < /dev/null
  if [ "$("$ADB" shell "ps -A -o NAME | grep -cx llama-server" < /dev/null | tr -d '\r')" != "0" ]; then
    say "$tag: STOP, a llama-server is still running"; exit 1
  fi
  say "$tag: start server (kv $kv)"
  "$ADB" shell "cd $BIN && nohup ./llama-server -m $BIN/model.gguf --host 127.0.0.1 --port $PORT -t 4 -c 4096 -np 1 \
    --jinja --no-webui --load-mode none -ctk $kv -ctv $kv --slot-save-path $BIN/slots/ > $BIN/$tag.log 2>&1 < /dev/null &" \
    < /dev/null > /dev/null 2>&1 &
  for i in $(seq 60); do
    "$ADB" shell "grep -q 'listening on' $BIN/$tag.log 2>/dev/null" < /dev/null && break
    sleep 2
  done
  say "$tag: loaded: $("$ADB" shell "grep -m1 -E 'general.name' $BIN/$tag.log; grep -m1 -E 'K \\(|type_k' $BIN/$tag.log" < /dev/null | tr -d '\r' | tr '\n' ' ' | cut -c1-300)"
  "$PY" -I 'C:\koffient-work\opt\cache_probe.py' "http://127.0.0.1:$PORT" "$(cygpath -w "$OUT/cache2.jsonl")" "$tag" >> "$LOG" 2>&1
  say "$tag: probe exit $?"
  "$ADB" shell "pkill -x llama-server; sleep 2; ls -l $BIN/slots/; rm -f $BIN/slots/*" < /dev/null >> "$LOG" 2>&1
  "$ADB" pull "$BIN/$tag.log" "$(cygpath -w "$OUT/$tag-server2.log")" >/dev/null
  sleep 60
}
for pair in "granite C:\\koffient-work\\release\\granite-4.0-1b-q4_k_m\\granite-4.0-1b-q4_k_m.gguf" \
            "qwen3 C:\\koffient-work\\release\\qwen3-0.6b-q4_k_m-tied\\qwen3-0.6b-tied-q4_k_m-tied.gguf"; do
  name=${pair%% *}; m=${pair#* }
  say "$name: push model"
  "$ADB" push "$m" "$BIN/model.gguf" >/dev/null
  probe "$name-f16" f16
  probe "$name-q8_0" q8_0
  "$ADB" shell "rm -f $BIN/model.gguf" < /dev/null
done
"$ADB" forward --remove tcp:$PORT
"$ADB" shell "pkill -x llama-server; rm -rf $BIN" < /dev/null
say "storage after cleanup: $("$ADB" shell df -h /data < /dev/null | tr -d '\r' | tail -1)"
say "all done"
