# Models Training Pipeline

This folder contains scripts for training our different models (detection, segmentation, and OBB) locally or on Slurm.

---

# Quick Start

## 1. Create a virtual environment

Always do this once before running anything:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pyyaml
```

> The script will automatically install model-specific dependencies later.

---

## 2. Set up your Roboflow API key

Training automatically downloads datasets from Roboflow. You must provide a **ROBOFLOW_API_KEY**.

Get your key from Roboflow -> Workspace -> Settings -> API Keys

You can set it in two ways:

### Option A — Environment variable
```bash
export ROBOFLOW_API_KEY=your_key_here
```
### Option B — `.env` file
Create a file called `.env` in the folder and add:
```bash
ROBOFLOW_API_KEY=your_key_here
```

> **Do NOT commit .env to git.**

## 3. Run training

Run:

```bash
python3 run.py
```

The script will automatically:

- Read `config.yaml`
- Choose the selected task (detection / segmentation / obb)
- Install the correct requirements file
- Run `train.py`

You do **not** need to manually install dependencies for each model.

---

## 4. Change settings

All settings are in:

```bash
config.yaml
```

You can change:

### Run mode

```yaml
run:
  mode: "local"   # or "slurm"
```

### Model type

```yaml
run:
  task: "detection"   # segmentation / detection / obb
```

### Training parameters

Each model has its own section:

```yaml
models:
  detection:
    model: "yolov8m.pt"
    epochs: 200
    imgsz: 640
```

### Slurm settings

```yaml
slurm:
  partition: "GPUQ"
  gpus: 1
  time: "1:00:00"
```

---

# How It Works

## run.py

This script:

1. Loads `config.yaml`
2. Picks the selected task
3. Installs dependencies from the task's requirements file
4. Runs training locally or submits a Slurm job

Example requirements mapping in config:

```yaml
models:
  detection:
    requirements: "yolo_object_detection/requirements.txt"
```

So switching task automatically switches dependencies.

---

## Local mode

```bash
python3 run.py --mode local
```

- Installs dependencies with pip
- Runs training directly

---

## Slurm mode

```bash
python3 run.py --mode slurm
```

- Creates a job script
- Installs dependencies on the cluster
- Runs training on GPU nodes

Make sure your Slurm account settings are correct in `config.yaml`.

---

# Examples

Run detection locally:

```bash
python3 run.py --task detection
```

Run segmentation on Slurm:

```bash
python3 run.py --mode slurm --task segmentation
```

# Checking SLurm Jobs

After submitting a job like:

```bash
python3 run.py
Submitted batch job 24063371
```

For each job, set the job ID (to debug it more easily):
```bash
export SLURM_JOB=
```

Now you can copy-paste the commands below.

## Check if job is running or pending

```bash
squeue -u $USER
```

Check a specific job:

```bash
squeue -j $SLURM_JOB
```

## See detailed job info

```bash
scontrol show job $SLURM_JOB
```

Shows:
- allocated GPU/CPU
- node name
- job state
- memory request

## Watch job log live
Find log file:
```bash
ls *$SLURM_JOB*.out
```

Watch it update:
```bash
tail -f *$SLURM_JOB*.out
```

## Check finished job stats

```bash
sacct -j $SLURM_JOB --format=JobID,State,Elapsed,MaxRSS,AllocGRES
```

Shows runtime, memory usage, GPU usage.

## Cancel a job

```bash
scancel $SLURM_JOB
```
