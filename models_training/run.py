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
    assert cfg["run"]["task"] in cfg["models"], (
        f"Task {cfg['run']['task']} not in models"
    )


def run_local(cfg):
    print("Running locally")

    task = cfg["run"]["task"]
    req = cfg["models"][task].get("requirements")

    if req:
        print(f"Installing requirements from {req}")
        subprocess.run(["pip", "install", "-r", req], check=True)

    subprocess.run(
        ["python3", cfg["run"]["train_script"], "--config", CONFIG_PATH],
        check=True,
    )


def submit_slurm(cfg):
    print("Submitting Slurm job")

    Path("logs").mkdir(exist_ok=True)

    template = Path(TEMPLATE_PATH).read_text()
    s = cfg["slurm"]

    requirements = cfg["models"][cfg["run"]["task"]].get("requirements", "")

    script = template.format(
        job_name=s["job_name"],
        partition=s["partition"],
        gpus=s["gpus"],
        cpus=s["cpus"],
        mem=s["mem"],
        time=s["time"],
        venv_path=s["venv_path"],
        train_script=cfg["run"]["train_script"],
        config=CONFIG_PATH,
        requirements=requirements,
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
