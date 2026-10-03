#!/bin/bash
# Submit Gemma 4 Fine-Tuning Job via gcloud CLI
set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
REGION="us-central1"
BUCKET_NAME="your-gcs-bucket-name"
MODEL_ID="google/gemma-4-31B-it"

# Choose accelerator configuration:
# For A100 80GB:
MACHINE_TYPE="a2-ultragpu-1g"
ACCELERATOR_TYPE="NVIDIA_A100_80GB"
ACCELERATOR_COUNT=1

# For H100 80GB (uncomment to use):
# MACHINE_TYPE="a3-highgpu-8g"
# ACCELERATOR_TYPE="NVIDIA_H100_80GB"
# ACCELERATOR_COUNT=8

DISPLAY_NAME="gemma4-31b-qlora-$(date +%s)"
OUTPUT_DIR="gs://${BUCKET_NAME}/output-${DISPLAY_NAME}"
CONTAINER_URI="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest"

echo "=========================================================="
echo "Submitting Vertex AI Custom Job: ${DISPLAY_NAME}"
echo "Project:      ${PROJECT_ID}"
echo "Accelerator:  ${ACCELERATOR_TYPE} x ${ACCELERATOR_COUNT}"
echo "Output:       ${OUTPUT_DIR}"
echo "=========================================================="

# 1. Upload training script to GCS staging
gcloud storage cp scripts/gemma4_31B_train.py "gs://${BUCKET_NAME}/scripts/gemma4_31B_train.py"

# 2. Submit Custom Job
cat <<EOF > /tmp/custom_job_spec.json
{
  "displayName": "${DISPLAY_NAME}",
  "jobSpec": {
    "workerPoolSpecs": [
      {
        "machineSpec": {
          "machineType": "${MACHINE_TYPE}",
          "acceleratorType": "${ACCELERATOR_TYPE}",
          "acceleratorCount": ${ACCELERATOR_COUNT}
        },
        "replicaCount": "1",
        "containerSpec": {
          "imageUri": "${CONTAINER_URI}",
          "command": [
            "bash",
            "-c",
            "pip install --quiet trl peft transformers bitsandbytes datasets accelerate && python3 /gcs/${BUCKET_NAME}/scripts/gemma4_31B_train.py"
          ],
          "env": [
            {"name": "MODEL_ID", "value": "${MODEL_ID}"},
            {"name": "AIP_MODEL_DIR", "value": "${OUTPUT_DIR}"},
            {"name": "HF_TOKEN", "value": "${HF_TOKEN}"}
          ]
        }
      }
    ],
    "baseOutputDirectory": {
      "outputUriPrefix": "${OUTPUT_DIR}"
    }
  }
}
EOF

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --config=/tmp/custom_job_spec.json

echo "Job submitted successfully!"
