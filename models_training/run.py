#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import yaml

CONFIG_PATH = "config.yaml"
TEMPLATE_PATH = "job_template.slurm"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def validate_config(cfg):
    task = cfg["run"]["task"]
    assert task in cfg["models"], f"Task {task} not in models"


def get_requirements(cfg):
    task = cfg["run"]["task"]
    return cfg["models"][task]["requirements"]


def install_requirements(cfg):
    req_file = get_requirements(cfg)
    mode = cfg["run"]["mode"]

    print(f"Installing requirements from {req_file} (mode={mode})")

    cmd = ["python3", "-m", "pip", "install", "-r", req_file]

    if mode == "slurm":
        cmd.insert(4, "--user")

    subprocess.run(cmd, check=True)


def run_local(cfg):
    print("Running locally")

    install_requirements(cfg)

    subprocess.run(
        ["python3", cfg["run"]["train_script"], "--config", CONFIG_PATH],
        check=True,
    )


def submit_slurm(cfg):
    print("Submitting Slurm job")

    Path("logs").mkdir(exist_ok=True)

    template = Path(TEMPLATE_PATH).read_text()
    s = cfg["slurm"]

    req_file = get_requirements(cfg)

    script = template.format(
        job_name=s["job_name"],
        account=s["account"],
        partition=s["partition"],
        nodes=s["nodes"],
        ntasks=s["ntasks"],
        cpus=s["cpus"],
        mem=s["mem"],
        time=s["time"],
        gpu_type=s["gpu_type"],
        gpus=s["gpus"],
        constraint=s["constraint"],
        train_script=cfg["run"]["train_script"],
        config=CONFIG_PATH,
        requirements=req_file,
    )

    Path("job.slurm").write_text(script)

    subprocess.run(["sbatch", "job.slurm"], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "slurm"])
    parser.add_argument("--task", choices=["segmentation", "detection", "obb"])
    args = parser.parse_args()

    cfg = load_config()

    if args.mode:
        cfg["run"]["mode"] = args.mode
    if args.task:
        cfg["run"]["task"] = args.task

    validate_config(cfg)

    if cfg["run"]["mode"] == "local":
        run_local(cfg)
    else:
        submit_slurm(cfg)


if __name__ == "__main__":
    main()
