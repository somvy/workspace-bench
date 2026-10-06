# full no-API check of a port branch: logit-lens readouts for every family on the branch's default model (2 GPUs), raw-text form, judge dry run
R=$(cd "$(dirname "$0")/../.." && pwd)
cd $R && export PYTHONPATH=$R/src HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/workspace-bench/.venv/bin/python; O=${O:-/workspace/wsb_full}; mkdir -p $O/logs $O/readouts/logit_lens $O/readouts/raw
FAMS=$($PY -c "from wsbench.registry import FAMILIES, load_all; load_all(); print(' '.join(sorted(FAMILIES)))")
echo families $FAMS
i=0
for f in $FAMS; do
  g=$((i % 2)); i=$((i + 1))
  echo "$g $f"
done > $O/queue.txt
for g in 0 1; do
  { grep "^$g " $O/queue.txt | while read _ f; do CUDA_VISIBLE_DEVICES=$g $PY -m wsbench produce family=$f method=logit_lens out=$O/readouts/logit_lens/$f.jsonl > $O/logs/prod_$f.log 2>&1; echo "$f exit $?" >> $O/logs/prod_status.txt; done; } &
done
wait
$PY - <<PYEOF
import json
from pathlib import Path
O = Path("$O")
for f in sorted((O / "readouts/logit_lens").glob("*.jsonl")):
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    (O / "readouts/raw" / f.name).write_text("".join(json.dumps({k: v for k, v in r.items() if k not in ("tokens", "scores")} | {"samples": r["tokens"]}, ensure_ascii=False) + "\n" for r in rows))
    print("raw", f.name, len(rows))
PYEOF
$PY -m wsbench run all=True dry_run=True readouts_root=$O/readouts/raw out=$O/judged > $O/logs/run_dry.log 2>&1; echo "run exit $?" >> $O/logs/prod_status.txt
echo DONE > $O/done
