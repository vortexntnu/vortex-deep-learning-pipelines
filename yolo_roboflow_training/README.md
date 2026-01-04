### Running the SLURM job
1. SSH into the cluster (IDUN)
2. Navigate to the project directory
3. Submit the job:
```bash
sbatch train_yolo.sbatch
```

### Environment variables

The following environment variable must be set before running training:

```bash
export ROBOFLOW_API_KEY=<your_roboflow_api_key>
```