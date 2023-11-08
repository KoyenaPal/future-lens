import os


def write_sbatch_file(in_layer: int, next_token_skip: int, use_context: bool, context_id: int):
    file_name = "L{}CXT{}TK{}.sh".format(str(in_layer), str(context_id), str(next_token_skip))
    with open(os.path.join("./sbatch", file_name), "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=L{in_layer}CXT{context_id}\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --gpus-per-node=1\n")
        f.write("#SBATCH --mem=80GB\n")
        f.write("#SBATCH --time=40:00:00\n")
        f.write("\n\n")
        f.write("source /data/jiuding_sun/.bashrc\n")
        f.write("cd /data/jiuding_sun/PrefixLens\n")
        command = f"python test.py --in_layer {in_layer}"
        if use_context:
            command += f" --text_prefix 1 --context_idx {context_id}"
        
        command += f" --next_token_skip {next_token_skip}"
        f.write(command)
        f.close()

if __name__ == "__main__":
    for in_layer in range(28):

        for next_token_skip in [0, 1, 2]:
            write_sbatch_file(in_layer, next_token_skip, False, 4)
            
        """for context_id in [0, 1, 2, 3]:
            write_sbatch_file(in_layer, 0, True, context_id)"""
        
    for file_name in os.listdir("./sbatch"):
        if file_name.endswith(".sh"):
            os.system(f"sbatch ./sbatch/{file_name}")

