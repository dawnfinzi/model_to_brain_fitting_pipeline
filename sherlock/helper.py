"""
Credit to Eshed Margalit for original
"""


def write_boilerplate(f, jobname="fit", time="14:00:00", mem="128000"):
    """
    Writes skeleton of sbatch script
    Inputs
        f: an open file buffer
        jobname (str): what to call the job in the queue
        time (str): how long to let the job run
        mem (str): how much memory to request
    """
    f.write("#!/bin/bash\n")
    f.write("#\n")
    f.write("#set the job name (output file and error file)\n")
    f.write(f"#SBATCH --job-name={jobname}\n")
    f.write(f"#SBATCH --output={jobname}.out\n")
    f.write(f"#SBATCH --error={jobname}.error\n")
    f.write("#############\n")
    f.write(f"#SBATCH --time={time}\n")
    f.write("#############\n")
    f.write("#SBATCH -p owners,normal,hns,kalanit\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#############\n")
    f.write(f"#SBATCH --mem={mem}\n")
    f.write("#############\n")
    f.write("\n")
    f.write("module load python/3.9\n")


def write_boilerplate_parallel(
    f, jobname="fit", time="14:00:00", mem="256000", n_task_ids=33
):
    """
    Writes skeleton of sbatch script using array
    Inputs
        f: an open file buffer
        jobname (str): what to call the job in the queue
        time (str): how long to let the job run
        mem (str): how much memory to request
    """
    f.write("#!/bin/bash\n")
    f.write("#\n")
    f.write("#set the job name (output file and error file)\n")
    f.write(f"#SBATCH --job-name={jobname}\n")
    f.write(f"#SBATCH --output={jobname}.%A_%a.out\n")
    f.write(f"#SBATCH --error={jobname}.%A_%a.error\n")
    f.write("#############\n")
    f.write(f"#SBATCH --time={time}\n")
    f.write("#############\n")
    f.write("#SBATCH -p owners,normal,hns,kalanit\n")
    f.write("#SBATCH --nodes=1\n")
    f.write(f"#SBATCH --array=0-{n_task_ids}\n")
    f.write("#############\n")
    f.write(f"#SBATCH --mem={mem}\n")
    f.write("#############\n")
    f.write("\n")
    f.write("module load python/3.9\n")
