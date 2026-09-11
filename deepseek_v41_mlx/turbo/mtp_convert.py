"""Convert the three mtp-only shards of deepseek-ai/DeepSeek-V4.1-Flash into one MLX safetensors
file for the DSpark drafter: fp8 e4m3 (32x32 ue8m0) -> bf16, 'I8' experts (= packed fp4 e2m1 pairs,
ue8m0 per 32 along input, exactly like the main model's experts) -> bf16, experts stacked into
SwitchGLU [E, out, in] tensors, then MLX affine 8-bit g64 for the quantizable projections
(the converter's rules: wq_a/wq_b/wkv/wo_b/experts/markov embed+head; wo_a stays bf16,
hc/sink/norms/gate/confidence stay as-is). mtp.<s>.* -> stages.<s>.*"""
import json, struct, os, re, sys, time
import numpy as np, mlx.core as mx

from deepseek_v41_mlx.dequant import dequant_fp8, dequant_fp4
SRC = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-mtp-shards"); DST = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-dspark")
BITS, GS = 8, 64


def main():
    def raw_tensors(path):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
        for k, e in hdr.items():
            if k == "__metadata__": continue
            a, b = e["data_offsets"]; yield k, e["dtype"], e["shape"], mm[a:b]
    def to_mx(raw, dtype, shape):
        if dtype in ("F8_E4M3", "F8_E8M0", "I8"): return mx.array(np.array(raw)).reshape(shape)      # uint8 view
        if dtype == "BF16": return mx.array(np.array(raw).view(np.uint16)).view(mx.bfloat16).reshape(shape)
        if dtype == "F32": return mx.array(np.array(raw).view(np.float32)).reshape(shape)
        raise ValueError(dtype)
    t0 = time.time(); pend = {}; out = {}; qmods = {}; experts = {}
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".safetensors"))
    for f in files:
        for k, dt, sh, raw in raw_tensors(os.path.join(SRC, f)):
            pend[k] = (dt, sh, raw)
    print(f"[mtp] {len(pend)} raw tensors from {len(files)} shards", flush=True)
    def quant_target(name):
        base = name.rsplit(".", 2)
        return len(base) >= 2 and base[-2] in ("wq_a", "wq_b", "wkv", "wo_b", "gate_proj", "up_proj", "down_proj", "embed", "head")
    for k in sorted(pend):
        if k.endswith(".scale"): continue
        dt, sh, raw = pend[k]
        w = to_mx(raw, dt, sh)
        sk = k[:-len(".weight")] + ".scale" if k.endswith(".weight") else None
        if sk in pend:
            s = to_mx(*pend[sk][2:3][0:1] and (pend[sk][2], pend[sk][0], pend[sk][1]))
            w = dequant_fp4(w, s, mx.bfloat16) if (dt == "I8" and ".ffn.experts." in k) else dequant_fp8(w, s, mx.bfloat16)
        name = re.sub(r"^mtp\.(\d+)\.", r"stages.\1.", k)
        if re.search(r"(hc_\w+|attn_sink|gate\.bias(_vl)?)$", name): w = w.astype(mx.float32)
        m = re.match(r"(stages\.\d+\.ffn\.experts)\.(\d+)\.(w[123])\.weight$", name)
        if m:
            proj = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[m.group(3)]
            experts.setdefault(f"{m.group(1)}.{proj}.weight", {})[int(m.group(2))] = w
            continue
        out[name] = w
    for final, parts in experts.items():
        out[final] = mx.stack([parts[i] for i in range(len(parts))]); mx.eval(out[final])
    print(f"[mtp] dequantized in {time.time()-t0:.0f}s: {sum(v.size for v in out.values())/1e9:.2f} B params", flush=True)
    q = {}
    for name in list(out):
        w = out[name]
        if quant_target(name) and w.shape[-1] % GS == 0 and w.dtype == mx.bfloat16:
            wq, sc, bi = mx.quantize(w, group_size=GS, bits=BITS); base = name[:-len(".weight")]
            q[f"{base}.weight"] = wq; q[f"{base}.scales"] = sc; q[f"{base}.biases"] = bi
            qmods[base] = {"group_size": GS, "bits": BITS}; del out[name]
        else:
            q[name] = w
    mx.eval(q)
    mx.save_safetensors(os.path.join(DST, "dspark.safetensors"), q)
    json.dump({"bits": BITS, "group_size": GS, "modules": qmods}, open(os.path.join(DST, "quantization.json"), "w"), indent=1)
    nb = sum(v.nbytes for v in q.values())
    print(f"[mtp] saved {len(q)} tensors, {nb/1e9:.2f} GB, {len(qmods)} quantized modules, {time.time()-t0:.0f}s", flush=True)
    for n in sorted(q)[:8] + [x for x in sorted(q) if "stages.0." in x and "experts" not in x][:40:4]: print("   ", n, q[n].shape, q[n].dtype)


if __name__ == "__main__":
    main()
