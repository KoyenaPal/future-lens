import os


def write_sbatch_file(in_layer: int):
    file_name = "L{}T1-5.sh".format(str(in_layer))
    with open(os.path.join("./sbatch", file_name), "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=L{in_layer}T1-5\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --gpus-per-node=1\n")
        f.write("#SBATCH --mem=80GB\n")
        f.write("#SBATCH --time=40:00:00\n")
        f.write("\n\n")
        f.write("source /data/jiuding_sun/.bashrc\n")
        f.write("cd /data/jiuding_sun/PrefixLens\n")
        f.write(f"python train.py --in_layer {in_layer} --next_token_skip 1 2 3 4 5")
        f.close()

if __name__ == "__main__":
    for in_layer in range(0, 28):
        write_sbatch_file(in_layer)
    
    for file_name in os.listdir("./sbatch"):
        if file_name.endswith(".sh"):
            os.system(f"sbatch ./sbatch/{file_name}")

