import boto3
import datetime
import json

def lambda_handler(event, context):
    # 建立 SageMaker client
    sagemaker_client = boto3.client('sagemaker')

    # 從 event 取得項目名稱，預設為 "Retail"
    item = event.get("itemList", "Retail")

    # 配置 (請替換成你的真實值)
    role_arn = "<YOUR_SAGEMAKER_EXECUTION_ROLE_ARN>"
    image_uri = "<YOUR_ECR_IMAGE_URI>"
    model_data_url = f"s3://<YOUR_S3_BUCKET>/analysis_predicted/{item}/model.tar.gz"

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

    # 回傳結果
    return {
        "statusCode": 200,
        "body": json.dumps(f"Model created: {model_name}"),
        "ModelName": model_name
    }
