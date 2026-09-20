#!/usr/bin/env bash
# 對照微優化：舊行為 (empty_cache+tqdm) vs 新預設 (skip cache + TQDM_DISABLE + SV cache)
set -euo pipefail
cd /home/sky/code/GPT-SoVITS
INPUT=local_tts/output/_bench_timing_input.txt
OUT=local_tts/output/_bench_micro
mkdir -p "$OUT"

run_api() {
  local tag=$1
  shift
  pkill -f 'api_v2.py' 2>/dev/null || true
  sleep 2
  pkill -9 -f 'api_v2.py' 2>/dev/null || true
  pkill -9 -f 'local_tts.start_api' 2>/dev/null || true
  sleep 1
  env "$@" ./local_tts/start_api.sh --role 真人男 > "$OUT/api_${tag}.log" 2>&1 &
  for i in $(seq 1 90); do
    if curl -sf -o /dev/null http://127.0.0.1:9880/docs; then
      sleep 2
      return 0
    fi
    sleep 2
  done
  echo "API failed to start ($tag)" >&2
  return 1
}

bench_once() {
  local tag=$1
  ./local_tts/.venv/bin/python - <<PY
import json, time, sys
from pathlib import Path
sys.path.insert(0, "local_tts")
sys.path.insert(0, ".")
# use package
from local_tts.bench_sweep import SweepConfig, run_config
from local_tts.common import get_role_profile

raw = Path("$INPUT").read_text(encoding="utf-8").replace("\\n","").replace(" ","")
profile = get_role_profile("真人男")
cfg = SweepConfig(
    name="$tag",
    batch_size=64,
    max_text_length=1200,
    fragment_interval=0.01,
)
# warmup
run_config("http://127.0.0.1:9880", profile, raw[:400], SweepConfig(name="w", batch_size=40, max_text_length=800))
rows = []
for i in range(2):
    row = run_config("http://127.0.0.1:9880", profile, raw, SweepConfig(name=f"$tag-{i+1}", batch_size=64, max_text_length=1200, fragment_interval=0.01))
    rows.append(row)
    print(f"$tag run{i+1}: {row['chars_per_s']:.1f} c/s wall={row['wall_s']:.2f}s ok={row['ok']} err={row['error']}")
ok = [r for r in rows if r["ok"]]
avg = sum(r["chars_per_s"] for r in ok)/len(ok) if ok else 0
print(f"$tag AVG: {avg:.1f} c/s")
Path("$OUT/${tag}.json").write_text(json.dumps({"avg_cps": avg, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

export PYTHONPATH="/home/sky/code/GPT-SoVITS/local_tts/.venv/lib/python3.10/site-packages:/home/sky/code/GPT-SoVITS:/home/sky/code/GPT-SoVITS/local_tts:/home/sky/code/GPT-SoVITS/.venv/lib/python3.10/site-packages"

echo "=== A: old-like (EMPTY_CACHE=1, TQDM on) ==="
run_api old GPT_SOVITS_EMPTY_CACHE=1 TQDM_DISABLE=0
bench_once old

echo "=== B: micro (EMPTY_CACHE=0, TQDM off, SV cache in code) ==="
run_api micro GPT_SOVITS_EMPTY_CACHE=0 TQDM_DISABLE=1
bench_once micro

echo "=== DONE ==="
python3 - <<'PY'
import json
from pathlib import Path
old = json.loads(Path("local_tts/output/_bench_micro/old.json").read_text())
micro = json.loads(Path("local_tts/output/_bench_micro/micro.json").read_text())
a, b = old["avg_cps"], micro["avg_cps"]
print(f"old={a:.1f} c/s  micro={b:.1f} c/s  delta={(b/a-1)*100:+.1f}%" if a else "n/a")
PY
