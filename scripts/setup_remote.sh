#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/root/autodl-tmp/hf
mkdir -p /root/autodl-tmp/{hf,data,ckpt,results}
pip install -q --upgrade \
  "transformers>=4.51" "datasets>=2.19" accelerate \
  scikit-learn lightgbm pandas pyarrow tqdm pyyaml pytest
python - <<'PY'
import transformers, datasets, torch
print("transformers", transformers.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
