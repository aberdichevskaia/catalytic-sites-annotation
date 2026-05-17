with open("run.sh", "w") as f:
    for i in range(0, 960, 80):
        f.write("sleep 0.1h\n")
        f.write(f"sbatch --array={i}-{i+39} /home/iscb/wolfson/annab4/catalytic-sites-annotation/model_search/run_array.slurm\n")
        f.write(f"sbatch --array={i+40}-{i+79} /home/iscb/wolfson/annab4/catalytic-sites-annotation/model_search/run_array.slurm\n")