#!/usr/bin/env python3
"""Submit a Gemma 4 Fine-Tuning Custom Job to Google Cloud Vertex AI.

Supports targeting:
  - NVIDIA A100 80GB (machine_type='a2-ultragpu-1g', accelerator_type='NVIDIA_A100_80GB')
  - NVIDIA H100 80GB (machine_type='a3-highgpu-8g', accelerator_type='NVIDIA_H100_80GB')
  - NVIDIA L4 / A100 40GB for smaller models

Usage:
  python3 submit_vertex_job.py \
      --project-id YOUR_PROJECT_ID \
      --bucket-name YOUR_GCS_BUCKET \
      --accelerator A100 \
      --model-id google/gemma-4-31B-it \
      --dataset-gcs-path gs://YOUR_BUCKET/custom_data/training_data.json
"""

import argparse
import os
import sys
from google.cloud import aiplatform


ACCELERATOR_CONFIGS = {
    "A100": {
        "machine_type": "a2-ultragpu-1g",
        "accelerator_type": "NVIDIA_A100_80GB",
        "accelerator_count": 1,
    },
    "A100-40GB": {
        "machine_type": "a2-highgpu-1g",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 1,
    },
    "H100": {
        "machine_type": "a3-highgpu-8g",
        "accelerator_type": "NVIDIA_H100_80GB",
        "accelerator_count": 8,
    },
    "L4": {
        "machine_type": "g2-standard-16",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 1,
    },
}

DEFAULT_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest"


def main():
    parser = argparse.ArgumentParser(description="Submit Vertex AI Fine-Tuning Job for Gemma 4")
    parser.add_argument("--project-id", type=str, default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    parser.add_argument("--region", type=str, default="us-central1")
    parser.add_argument("--bucket-name", type=str, required=True, help="GCS bucket name for staging & output")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--accelerator", choices=["A100", "A100-40GB", "H100", "L4"], default="A100")
    parser.add_argument("--dataset-gcs-path", type=str, default="", help="Optional GCS path to training JSON")
    parser.add_argument("--container-uri", type=str, default=DEFAULT_CONTAINER_URI)
    parser.add_argument("--display-name", type=str, default="gemma4-31b-qlora-sft")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--no-wait", action="store_true", help="Submit job and return immediately without blocking")

    args = parser.parse_args()

    if not args.project_id:
        print("Error: --project-id is required (or set GOOGLE_CLOUD_PROJECT).", file=sys.stderr)
        sys.exit(1)

    accel_spec = ACCELERATOR_CONFIGS[args.accelerator]

    staging_bucket = f"gs://{args.bucket_name.replace('gs://', '')}"
    output_dir = f"{staging_bucket}/output-{args.accelerator}-{args.display_name}"

    print("=" * 70)
    print("SUBMITTING VERTEX AI CUSTOM TRAINING JOB")
    print("=" * 70)
    print(f"Project:         {args.project_id}")
    print(f"Region:          {args.region}")
    print(f"Model ID:        {args.model_id}")
    print(f"Accelerator:     {accel_spec['accelerator_type']} x {accel_spec['accelerator_count']}")
    print(f"Machine Type:    {accel_spec['machine_type']}")
    print(f"Output GCS:      {output_dir}")
    print(f"Container:       {args.container_uri}")

    # Initialize Vertex AI SDK
    aiplatform.init(
        project=args.project_id,
        location=args.region,
        staging_bucket=staging_bucket,
    )

    # Environment variables injected into the container
    env_vars = [
        {"name": "MODEL_ID", "value": args.model_id},
        {"name": "AIP_MODEL_DIR", "value": output_dir},
        {"name": "NUM_EPOCHS", "value": str(args.epochs)},
        {"name": "BATCH_SIZE", "value": str(args.batch_size)},
        {"name": "LEARNING_RATE", "value": str(args.learning_rate)},
        {"name": "LORA_R", "value": str(args.lora_r)},
        {"name": "LORA_ALPHA", "value": str(args.lora_alpha)},
    ]

    if args.dataset_gcs_path:
        env_vars.append({"name": "DATASET_GCS_PATH", "value": args.dataset_gcs_path})

    # Forward HF_TOKEN if present in environment
    if "HF_TOKEN" in os.environ:
        env_vars.append({"name": "HF_TOKEN", "value": os.environ["HF_TOKEN"]})

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": accel_spec["machine_type"],
                "accelerator_type": accel_spec["accelerator_type"],
                "accelerator_count": accel_spec["accelerator_count"],
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": args.container_uri,
                "command": [
                    "bash",
                    "-c",
                    "pip install --quiet trl peft transformers bitsandbytes datasets accelerate && python3 /gcs/"
                    + args.bucket_name.replace("gs://", "")
                    + "/scripts/gemma4_31B_train.py",
                ],
                "env": env_vars,
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=args.display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=output_dir,
    )

    print("\nTriggering CustomJob on Vertex AI...")
    job.run(sync=not args.no_wait)

    print("\nJob successfully dispatched!")
    print(f"Resource Name: {job.resource_name}")


if __name__ == "__main__":
    main()
