"""Disk guard. The data disk is only 60G; filling it up silently corrupts checkpoints."""
import shutil


class DiskSpaceError(RuntimeError):
    pass


def free_gb(path: str = "/root/autodl-tmp") -> float:
    return shutil.disk_usage(path).free / 2**30


def require_free_gb(need: float, path: str = "/root/autodl-tmp") -> None:
    """Raise immediately when free space is insufficient, instead of letting a run
    fail halfway through by filling up the disk."""
    have = free_gb(path)
    if have < need:
        raise DiskSpaceError(
            f"{path} has {have:.1f}G free, below the required {need:.1f}G. "
            f"Please clean up first: old checkpoints (keep only 1), raw Amazon files, HF cache."
        )
