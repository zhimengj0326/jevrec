"""W0 blocking items: confirm SWA, measure tokens/sec, and count the KuaiRand random-exposure sample size."""
import os, time, argparse, torch
from transformers import AutoConfig, AutoModelForCausalLM

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")

from jevrec.utils.disk import require_free_gb


def check_swa(model_id: str) -> bool:
    """Returns True if a sliding-window layer is present (this breaks the block-diagonal mask semantics)."""
    cfg = AutoConfig.from_pretrained(model_id)
    use_swa = bool(getattr(cfg, "use_sliding_window", False))
    window = getattr(cfg, "sliding_window", None)
    layer_types = getattr(cfg, "layer_types", None)
    print(f"  use_sliding_window={use_swa} sliding_window={window} layer_types={layer_types}")
    if layer_types is not None:
        return any("sliding" in str(t) for t in layer_types)
    return use_swa and window is not None


def measure_tokens_per_sec(model_id: str, seq_len: int, batch: int, steps: int = 12) -> float:
    """Real forward+backward throughput. The first 4 steps are discarded as warmup."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    model.gradient_checkpointing_disable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    ids = torch.randint(0, 1000, (batch, seq_len), device="cuda")
    elapsed = 0.0
    for i in range(steps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 4:
            elapsed += time.perf_counter() - t0
    return (steps - 4) * batch * seq_len / elapsed


def count_kuairand_random(data_dir: str) -> int:
    import pandas as pd, glob
    total = 0
    for f in glob.glob(os.path.join(data_dir, "*.csv")):
        for chunk in pd.read_csv(f, usecols=lambda c: c == "is_random", chunksize=500_000):
            if "is_random" in chunk:
                total += int(chunk["is_random"].sum())
    return total


if __name__ == "__main__":
    require_free_gb(5.0)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--seq-len", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--kuairand-dir", default="/root/autodl-tmp/data/KuaiRand-Pure/data")
    ap.add_argument(
        "--skip-throughput",
        action="store_true",
        help="Run only the SWA check (fetches config.json only, no weight download) and skip "
        "the throughput measurement. For models whose weights haven't been downloaded yet "
        "or aren't planned for download soon, e.g. larger models deferred in a staged rollout.",
    )
    a = ap.parse_args()

    print(f"=== SWA check: {a.model} ===")
    swa = check_swa(a.model)
    print(f"SWA_PRESENT: {swa}")

    if a.skip_throughput:
        print("=== Throughput measurement: skipped (--skip-throughput) ===")
        print("TOKENS_PER_SEC: SKIPPED (--skip-throughput)")
    else:
        print(f"=== Throughput measurement (seq_len={a.seq_len}, batch={a.batch}) ===")
        tps = measure_tokens_per_sec(a.model, a.seq_len, a.batch)
        print(f"TOKENS_PER_SEC: {tps:.0f}")
        print(f"  Converted: one {a.seq_len}-token training sample ≈ {a.seq_len / tps * 1000:.1f} ms")

    if os.path.isdir(a.kuairand_dir):
        n = count_kuairand_random(a.kuairand_dir)
        print(f"KUAIRAND_RANDOM_ROWS: {n}")
    else:
        print("KUAIRAND_RANDOM_ROWS: SKIPPED (data not downloaded)")
