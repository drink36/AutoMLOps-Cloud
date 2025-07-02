import argparse
import json
import os
from model import trainer_factory
from trainer import DataTrainer
def get_hyperparameters():
    if 'SAGEMAKER_HYPERPARAMETERS' in os.environ:
        return json.loads(os.environ['SAGEMAKER_HYPERPARAMETERS'])
    else:
        return {}

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='S3 or local path for CSV')
    
    # 訓練參數
    # 這些參數可以根據需要進行調整
    # length of the time series, number of epochs, batch size, etc.
    parser.add_argument('--time_series_length', type=int, default=30, help='Length of the time series for each customer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for the model')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for the model')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of LSTM layers')

    # 其他訓練參數
    parser.add_argument('--use_gpu', type=str, default="False", help='是否使用 GPU (True/False)')

    # model 參數
    parser.add_argument('--model_backend', choices=['xgboost','pytorch'], default='xgboost', help='模型後端選擇')


    # 上傳 sample 資料的旗標與 S3 參數
    parser.add_argument('--upload_sample_to_s3', type=str, default="False", help='是否上傳 sample 資料到 S3 (True/False)')
    parser.add_argument('--sample_s3_bucket', type=str, default="", help='S3 bucket for sample prediction data')
    parser.add_argument('--sample_s3_prefix', type=str, default="", help='S3 prefix/path for sample prediction data')
    
    # 上傳模型工件的旗標與 S3 參數
    parser.add_argument('--upload_model_to_s3', type=str, default="False", help='是否上傳模型工件到 S3 (True/False)')
    parser.add_argument('--model_s3_bucket', type=str, default="", help='S3 bucket for model artifact')
    parser.add_argument('--model_s3_prefix', type=str, default="", help='S3 prefix/path for model artifact')
    

    args = parser.parse_args()



    upload_sample = args.upload_sample_to_s3.lower() == "true"
    upload_model  = args.upload_model_to_s3.lower() == "true"

    # 驗證 S3 上傳時必須提供的參數
    if upload_sample and (args.sample_s3_bucket == "" or args.sample_s3_prefix == ""):
        raise ValueError("上傳 sample 資料到 S3 時，必須提供 sample_s3_bucket 與 sample_s3_prefix。")
    if upload_model and (args.model_s3_bucket == "" or args.model_s3_prefix == ""):
        raise ValueError("上傳模型工件到 S3 時，必須提供 model_s3_bucket 與 model_s3_prefix。")
    
    # 取得 SageMaker 預設模型輸出目錄，預設為本地當前目錄
    model_dir = os.environ.get('SM_MODEL_DIR', '.')

    file_paths = {
    "all": args.file_path,
    }
    
    dt = DataTrainer(
        file_paths=file_paths,
        sequence_length=args.time_series_length
    )
    
    dt.filter_pipeline()

    filtered_data,features_predict,time_series = dt.prepare_processed_data()
    mt = trainer_factory(
        args.model_backend,
        "train",                   # mode
        features_predict,          # 預測用特徵
        filtered_data,             # 處理過的資料
        target_col="OrderGapDays"  # 目標欄位
    )

    mt.run_pipeline(
        path=os.path.join(model_dir, "model_artifact.pkl"),
        to_s3=upload_model,
        s3_bucket=args.model_s3_bucket if upload_model else None,
        s3_prefix=args.model_s3_prefix if upload_model else None)
    
    mt.build_sample_prediction_data(
        output_path=os.path.join(model_dir, "sample_prediction.csv"),
        to_s3=upload_sample,
        s3_bucket=args.sample_s3_bucket if upload_sample else None,
        s3_prefix=args.sample_s3_prefix if upload_sample else None
    )

if __name__ == '__main__':
    main()

