# serve.py

import os
import io
import tarfile
import pickle
import pandas as pd
import json

from inference_trainer import InferenceTrainer

# load model
def model_fn(model_dir: str) -> InferenceTrainer:
    gz = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(gz):
        with tarfile.open(gz, "r:gz") as tar:
            tar.extractall(path=model_dir)

    pkl = os.path.join(model_dir, "model_artifact.pkl")
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"找不到 {pkl}")
    with open(pkl, "rb") as f:
        trainer = pickle.load(f)
    return InferenceTrainer(trainer)

# 直接使用model處理後的sample data
def input_fn(input_data, content_type) -> pd.DataFrame:
    """
    把 HTTP 请求的 body 解成 DataFrame。
    支持 CSV 或 JSON。
    """
    if isinstance(input_data, bytes):
        input_data = input_data.decode("utf-8")
    try:
        if "csv" in content_type.lower():
            # 若輸入內容全是空白，則返回空 DataFrame
            if not input_data.strip():
                return pd.DataFrame()
            # 嘗試讀取 CSV，如果有格式錯誤則跳過不正確的行
            try:
                stream = io.StringIO(input_data)
                df = pd.read_csv(stream, on_bad_lines='skip')
                return df
            except SystemExit as e:
                return pd.DataFrame()
        elif "json" in content_type.lower():
            return json.loads(input_data)
        else:
            raise ValueError("Unsupported content type: " + content_type)
    except Exception as e:
        raise ValueError("Error parsing input data: " + str(e))

# 使用inferenceTrainer的predict方法
def predict_fn(data: pd.DataFrame, trainer: InferenceTrainer) -> pd.DataFrame:
    """
    调用 trainer.predict(...) 得到带 predicted_next_dt 的 DataFrame。
    如果没有 cluster 列，就把所有行都当到第一个模型里。
    """
    df = data.copy()
    return trainer.predict(df)


def output_fn(prediction: pd.DataFrame, accept: str) -> str:
    """
    把上一步的 DataFrame 格式化成 CSV 或 JSON 字符串返回给客户端。
    """
    if "csv" in accept.lower():
        buf = io.StringIO()
        prediction.to_csv(buf, index=False)
        return buf.getvalue()
    return prediction.to_json(orient="records")
