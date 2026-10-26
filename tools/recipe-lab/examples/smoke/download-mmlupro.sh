#!/bin/bash
# Download the MMLU-Pro dataset snapshot into ~/jinghaoyu/dataset/mmlu-pro.
# Usage: nohup bash ~/jinghaoyu/sr-build/scripts/download-mmlupro.sh \
#   > ~/jinghaoyu/recipe-lab-workspace/logs/dataset-mmlupro-download.log 2>&1 &
set -u
export HF_HOME="$HOME/jinghaoyu/hf-home"
mkdir -p "$HOME/jinghaoyu/dataset"
"$HOME/jinghaoyu/vllm028-env/bin/python" - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("TIGER-Lab/MMLU-Pro", repo_type="dataset", local_dir="/home/david_cwq/jinghaoyu/dataset/mmlu-pro")
print("DATASET-DONE", p, flush=True)
PYEOF
