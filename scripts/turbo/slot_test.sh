#!/bin/zsh
# big prompt A, tiny prompt B, then A + follow-up: the third must reuse A's slot.
P=${V41_PORT:-8001}; U="http://127.0.0.1:$P/v1/chat/completions"
BIG=$(python3 -c "import json; print(json.dumps(open('$HOME/qwen/LOCAL-DEEPSEEK-SETUP.md').read()[:12000]))")
req() { curl -s -m 900 $U -H 'Content-Type: application/json' -d "$1" | python3 -c "import sys,json; r=json.load(sys.stdin); print('  ->', repr(r['choices'][0]['message']['content'][:100]), r['usage'])"; }
echo "=== A (big)";  req "{\"model\":\"DeepSeek-V4.1-Flash\",\"max_tokens\":30,\"messages\":[{\"role\":\"user\",\"content\":$BIG},{\"role\":\"user\",\"content\":\"Summarize the above in one sentence.\"}]}"
echo "=== B (tiny)"; req '{"model":"DeepSeek-V4.1-Flash","max_tokens":10,"messages":[{"role":"user","content":"Say hi."}]}'
echo "=== A + follow-up (must reuse)"; req "{\"model\":\"DeepSeek-V4.1-Flash\",\"max_tokens\":30,\"messages\":[{\"role\":\"user\",\"content\":$BIG},{\"role\":\"user\",\"content\":\"Summarize the above in one sentence.\"},{\"role\":\"assistant\",\"content\":\"It is a setup log.\"},{\"role\":\"user\",\"content\":\"Name one measured number from it.\"}]}"
curl -s http://127.0.0.1:$P/health; echo
