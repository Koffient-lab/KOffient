"""Render the tool-calling TTFT prompt with each model's own chat template (as the SDK will), via llama-server
/apply-template. The hand-written ttft_tools_en.txt pretty-prints the tool JSON; the templates insert it with
`tojson` (one line), so the real prompt is shorter.

usage: python -I render_prompts.py <server url> <out file> <question variant: q1|q2>
"""
import json
import pathlib
import sys
import urllib.request

url, out, variant = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
src = pathlib.Path(__file__).with_name("tools_q1.txt").read_text(encoding="utf-8")
tools = json.loads(src[src.index("["):src.rindex("]") + 1])
questions = {
    "q1": src[src.rindex("User: ") + 6:src.rindex("\n\nAssistant:")],
    "q2": "How much is 250 US dollars in euros right now?",
}
body = {
    "messages": [
        {"role": "system", "content": "You are an assistant inside a mobile app. Call a function only when it is "
                                      "needed; otherwise answer in plain text. Today is Thursday, 12 March 2026, "
                                      "and the user is in Berlin."},
        {"role": "user", "content": questions[variant]},
    ],
    "tools": [{"type": "function", "function": t} for t in tools],
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(url + "/apply-template", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
prompt = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt"]
out.write_text(prompt, encoding="utf-8", newline="\n")
print(out.name, len(prompt), "chars")
