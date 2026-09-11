#!/bin/zsh
# Smoke tests against the v41 server: non-stream, stream, tool call, prefix reuse.
P=${V41_PORT:-8001}; U="http://127.0.0.1:$P/v1/chat/completions"; H='-H Content-Type:application/json -H Authorization:Bearer_v41-local'
echo "=== models"; curl -s http://127.0.0.1:$P/v1/models; echo
echo "=== non-stream"; curl -s -m 600 $U -H 'Content-Type: application/json' -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"In one sentence, what is a B-tree?"}],"max_tokens":80}' | python3 -c "import sys,json; r=json.load(sys.stdin); print(json.dumps(r['choices'][0]['message'])[:400]); print('usage', r['usage'])"
echo "=== stream (same prefix + follow-up -> prefix reuse)"; curl -s -N -m 600 $U -H 'Content-Type: application/json' -d '{"model":"DeepSeek-V4.1-Flash","stream":true,"messages":[{"role":"user","content":"In one sentence, what is a B-tree?"},{"role":"assistant","content":"A B-tree is a balanced search tree with wide nodes."},{"role":"user","content":"And a B+tree? One sentence."}],"max_tokens":80}' | python3 -c "
import sys,json
txt=''; fin=None
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('data: ') or line=='data: [DONE]': continue
    ch=json.loads(line[6:]); d=ch['choices'][0]['delta']; txt+=d.get('content','') or ''; fin=ch['choices'][0]['finish_reason'] or fin
    if 'usage' in ch: print('usage', ch['usage'])
print('streamed text:', repr(txt[:300])); print('finish:', fin)"
echo "=== tool call"; curl -s -m 600 $U -H 'Content-Type: application/json' -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"system","content":"You are a coding agent. Use tools when needed."},{"role":"user","content":"What is in /etc/hosts? Use the read_file tool."}],"tools":[{"type":"function","function":{"name":"read_file","description":"Read a file from disk","parameters":{"type":"object","properties":{"path":{"type":"string","description":"absolute path"}},"required":["path"]}}}],"max_tokens":200}' | python3 -c "import sys,json; r=json.load(sys.stdin); m=r['choices'][0]['message']; print('finish', r['choices'][0]['finish_reason']); print('content', repr(m.get('content')), 'tool_calls', json.dumps(m.get('tool_calls')))"
echo "=== thinking (reasoning_effort)"; curl -s -m 600 $U -H 'Content-Type: application/json' -d '{"model":"DeepSeek-V4.1-Flash","reasoning_effort":"low","messages":[{"role":"user","content":"Is 91 prime? Answer yes or no with a one-line reason."}],"max_tokens":400}' | python3 -c "import sys,json; r=json.load(sys.stdin); m=r['choices'][0]['message']; print('reasoning:', repr((m.get('reasoning_content') or '')[:200])); print('content:', repr(m.get('content'))[:300])"
