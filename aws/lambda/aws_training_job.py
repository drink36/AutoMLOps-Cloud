import json
import boto3
import datetime

def lambda_handler(event, context):
    # 建立 SageMaker client
    sagemaker_client = boto3.client('sagemaker')
    
    # 產生當下時間的字串 (格式：YYYYMMDDHHMMSS)
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    job_name = f"training-job-{current_time}"
    
    # 呼叫 create_training_job API 建立 Training Job
    response = sagemaker_client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": "<YOUR_ECR_IMAGE_URI>",
            "TrainingInputMode": "File"
        },
        Environment={
            "MODE": "train",
            "SAGEMAKER_PROGRAM": "train.py",
            "file_path": "<YOUR_S3_INPUT_URI>",
            "time_series_length": "15",
            "model_backend": "pytorch",
            "upload_sample_to_s3": "True",
            "sample_s3_bucket": "<YOUR_S3_BUCKET>",
            "sample_s3_prefix": "<YOUR_S3_SAMPLE_PREFIX>",
            "upload_model_to_s3": "True",
            "model_s3_bucket": "<YOUR_S3_BUCKET>",
            "model_s3_prefix": "<YOUR_S3_MODEL_PREFIX>"
        },
        RoleArn="<YOUR_SAGEMAKER_EXECUTION_ROLE_ARN>",
        OutputDataConfig={
            "S3OutputPath": "<YOUR_S3_OUTPUT_URI>"
        },
        ResourceConfig={
            "InstanceType": "ml.g4dn.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 600
        }
    )
    
    return {
        "statusCode": 200,
        "body": json.dumps(f"Training job {job_name} started!")
    }
