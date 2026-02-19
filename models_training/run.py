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

    subprocess.run(
        ["python3", cfg["run"]["train_script"], "--config", CONFIG_PATH],
        check=True,
    )


def submit_slurm(cfg):
    print("Submitting Slurm job")

    venv = ensure_venv(cfg)

    Path("logs").mkdir(exist_ok=True)

    template = Path(TEMPLATE_PATH).read_text()
    s = cfg["slurm"]

    script = template.format(
        job_name=s.get("job_name", "yolo-train"),
        account=s.get("account", ""),
        partition=s.get("partition", "GPUQ"),
        nodes=s.get("nodes", 1),
        ntasks=s.get("ntasks", 1),
        cpus=s.get("cpus", 8),
        mem=s.get("mem", "32G"),
        time=s.get("time", "1:00:00"),
        gpu_type=s.get("gpu_type", "a100"),
        gpus=s.get("gpus", 1),
        constraint=s.get("constraint", ""),
        train_script=cfg["run"]["train_script"],
        config=CONFIG_PATH,
        venv_path=str(venv.resolve()),
    )

    Path("job.slurm").write_text(script)

    subprocess.run(["sbatch", "job.slurm"], check=True)


def ensure_venv(cfg):
    task = cfg["run"]["task"]
    req = cfg["models"][task]["requirements"]

    venv = Path("venvs") / task

    if not (venv / "bin/python").exists():
        print(f"Creating venv for {task}")
        subprocess.run(["bash", "setup_venv.sh", task, req], check=True)
    else:
        print(f"Using existing venv: {venv}")

    return venv


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
