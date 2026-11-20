#!/usr/bin/env bash
# The chosen configuration on the phone: Granite q4_0-emb-q4_0-imat-ffn_down-q4_0 and Qwen3 q4_k_m-tied,
# KV cache q8_0, n_ctx 4096, six-tool request, load without mmap, 4 threads; 2 interleaved rounds. Then the prompt
# cache with the model resident (llama-server) for the new Granite file, KV q8_0 and f16.
set -u
export ANDROID_HOME='C:\Android' PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1
ADB=/c/Android/platform-tools/adb.exe
PY=/c/koffient-work/venv/Scripts/python.exe
LOG=/c/koffient-work/results/logs/phone-final.log
OUT=/c/koffient-work/results/raw/phone-cache
G='C:\koffient-work\models\granite-4.0-1b\6a7381b\granite-4.0-1b-q4_0-emb-q4_0-imat-ffn_down-q4_0.gguf'
Q='C:\koffient-work\release\qwen3-0.6b-q4_k_m-tied\qwen3-0.6b-tied-q4_k_m-tied.gguf'
BIN=/data/local/tmp/koffient/cache
PORT=8899
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
say "start"
for round in 1 2; do
  for m in "$G" "$Q"; do
    say "round $round: $(basename "$m")"
    "$PY" -I -m koffient_tooling bench run --work-dir 'C:\koffient-work' --build android-arm64-dotprod --model "$m" \
      --load-mode none --threads 4 --skip-cold --timeout 1800 --n-ctx 4096 --kv-cache-type q8_0 \
      --prompt-file ttft_tools_en.txt --label "phone-final-r$round" >> "$LOG" 2>&1
    say "done (exit $?)"
    "$ADB" shell "rm -rf /data/local/tmp/koffient/runner/models" < /dev/null
    sleep 60
  done
done

# Prompt cache, model resident (same procedure as phone-cache2.sh).
"$ADB" shell "pkill -x llama-server; mkdir -p $BIN/slots" < /dev/null
"$ADB" push 'C:\koffient-work\build\android-arm64-dotprod\bin\llama-server' "$BIN/llama-server" >/dev/null
"$ADB" shell "chmod 755 $BIN/llama-server" < /dev/null
"$ADB" push "$G" "$BIN/model.gguf" >/dev/null
"$ADB" forward tcp:$PORT tcp:$PORT
for kv in q8_0 f16; do
  tag="granite-q40emb4-$kv"
  "$ADB" shell "pkill -x llama-server; sleep 2" < /dev/null
  say "$tag: start server"
  "$ADB" shell "cd $BIN && nohup ./llama-server -m $BIN/model.gguf --host 127.0.0.1 --port $PORT -t 4 -c 4096 -np 1 \
    --jinja --no-webui --load-mode none -ctk $kv -ctv $kv --slot-save-path $BIN/slots/ > $BIN/$tag.log 2>&1 < /dev/null &" \
    < /dev/null > /dev/null 2>&1 &
  "$PY" -I 'C:\koffient-work\opt\cache_probe.py' "http://127.0.0.1:$PORT" "$(cygpath -w "$OUT/cache3.jsonl")" "$tag" >> "$LOG" 2>&1
  say "$tag: probe exit $?"
  "$ADB" shell "pkill -x llama-server; sleep 2; rm -f $BIN/slots/*" < /dev/null
  "$ADB" pull "$BIN/$tag.log" "$(cygpath -w "$OUT/$tag-server.log")" >/dev/null
  sleep 60
done
"$ADB" forward --remove tcp:$PORT
"$ADB" shell "pkill -x llama-server; rm -rf $BIN" < /dev/null
say "all done"
