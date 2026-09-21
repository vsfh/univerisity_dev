"""Run only the repaired retrieval baselines in a separate output directory."""
import sys

from baseline_suite import main


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Usage: bash run_clip_siglip2.sh (no arguments)")
    raise SystemExit(main(
        ["retrieval_clip", "retrieval_siglip"],
        output_name="clip_siglip2_recheck",
    ))
