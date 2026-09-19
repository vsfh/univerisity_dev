"""Run each existing heatmap experiment across three GPUs together."""
import study

if __name__ == "__main__":
    study.main(data_parallel_gpus=3)
