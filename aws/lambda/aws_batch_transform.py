import boto3
import datetime
import json

def lambda_handler(event, context):
    # 建立 SageMaker client
    sagemaker_client = boto3.client('sagemaker')

    # 配置（請替換為您的真實值）
    role_arn = "<YOUR_SAGEMAKER_EXECUTION_ROLE_ARN>"
    image_uri = "<YOUR_ECR_IMAGE_URI>"
    model_data_url = "<YOUR_S3_MODEL_DATA_URL>"

    # 產生唯一的模型名稱（包含時間戳記）
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"my-inference-model-{timestamp}"

    # 建立 SageMaker Model
    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "mode": "inference",
                "sagemaker_program": "inference.py"
            }
        },
        ExecutionRoleArn=role_arn
    )
    print("Model created:", model_name)

    # 產生唯一的 Batch Transform Job 名稱
    transform_job_name = f"batch-transform-job-{timestamp}"
    input_s3_uri = "<YOUR_S3_INPUT_URI>"
    output_s3_uri = "<YOUR_S3_OUTPUT_URI>"

    # 建立 Batch Transform Job
    sagemaker_client.create_transform_job(
        TransformJobName=transform_job_name,
        ModelName=model_name,
        TransformInput={
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_s3_uri
                }
            },
            "ContentType": "text/csv",
            "SplitType": "None"
        },
        TransformOutput={
            "S3OutputPath": output_s3_uri
        },
        TransformResources={
            "InstanceType": "ml.g4dn.xlarge",
            "InstanceCount": 1
        }
    )
    print("Transform job created:", transform_job_name)

    return {
        "statusCode": 200,
        "body": json.dumps(f"Transform job {transform_job_name} started!")
    }

