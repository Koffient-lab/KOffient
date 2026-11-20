# Experiment scripts

Scripts of the experiments described in [`docs/models/METHODOLOGY.md`](../../docs/models/METHODOLOGY.md), kept as
they were run. They assume the work folder `C:\koffient-work` (see [`tooling/README.md`](../README.md)), adb at
`C:\Android\platform-tools`, Git Bash on Windows and one Android phone connected over USB with debugging on. Results
go to `C:\koffient-work\results\` (runs.jsonl, quality.jsonl, logs/, raw/).

## phone/

| Script | What it measures |
|---|---|
| `phone-check.sh` | The MVP files on the phone: checks the CPU (dotprod required), picks the big cores, 3 interleaved rounds of `bench run` (short prompt and six-tool request), then n_ctx 4096 for memory |
| `phone-opt.sh` + `opt_device.sh` | Thread scaling 1-4 on the big cores, micro-batch x flash attention, `--poll` / `--cpu-strict` / `--prio`, prompt cache through `llama-completion --prompt-cache`; then memory runs (KV q8_0, micro-batch 128, 4-bit token table) |
| `phone-cache2.sh` + `cache_probe.py` | Prompt cache with the model resident: `llama-server` on the phone, requests through the model's chat template, cold / warm / slot save-restore |
| `phone-threads.sh` | 4/5/6/8 threads without a core mask (little cores included) |
| `phone-profile.sh` | `simpleperf` profile of prompt and generation, per thread and per function |
| `burn-test.sh` | Core clocks under 1 and 4 busy threads (the ~73% clock ceiling) |
| `phone-emb4.sh` | Granite Q4_0 with the token table in Q6_K / Q4_0 / Q4_K: speed and memory |
| `phone-final.sh` | The chosen configuration (KV q8_0, n_ctx 4096, six-tool request) and its prompt cache |
| `render_prompts.py` | Renders the six-tool request with a model's own chat template through `llama-server` |
| `tools_q1.txt`, `tools_q2.txt` | The six-tool prompt and the same prefix with another question |
| `summarize_opt.py` | Summary of the `phone-opt.sh` output |

## quality/

| Script | What it measures |
|---|---|
| `granite-emb4-cpu.sh`, `granite-emb4-cpu2.sh` | Granite Q4_0 candidate against the same recipe with a 4-bit token table, fast set on the CPU, task by task in turn, for a pairwise comparison |
