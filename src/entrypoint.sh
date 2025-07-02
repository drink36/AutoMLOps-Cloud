#!/usr/bin/env bash
set -e

case "${MODE}" in
  train)
    echo "[ENTRYPOINT] entry training mode"
    exec python train.py \
      --file_path                "${file_path}" \
      --time_series_length       "${time_series_length}" \
      --model_backend            "${model_backend}" \
      --upload_sample_to_s3      "${upload_sample_to_s3}" \
      --sample_s3_bucket         "${sample_s3_bucket}" \
      --sample_s3_prefix         "${sample_s3_prefix}" \
      --upload_model_to_s3       "${upload_model_to_s3}" \
      --model_s3_bucket          "${model_s3_bucket}" \
      --model_s3_prefix          "${model_s3_prefix}"
    ;;
  inference|*)
    echo "[ENTRYPOINT] entry inference mode"
    exec gunicorn -b 0.0.0.0:8080 app:app
    ;;
esac
