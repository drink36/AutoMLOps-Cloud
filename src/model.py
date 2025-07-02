import abc
import pickle
import os
import tarfile
import boto3


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ====== Base Trainer ======
class BaseModelTrainer(abc.ABC):
    def __init__(self, features_predict, filtered_data, target_col):
        self.features_predict = features_predict
        self.filtered_data = filtered_data
        self.target_col = target_col
        self.model = None
        self.scaler = None
    def run_pipeline(self, *args, **kwargs):
        self.data_preprocessing()
        self.train()
        self.save_model(*args, **kwargs)

    # 前處理完成檔案存放 base class 接口
    def build_sample_prediction_data(
        self,
        output_path: str = "sample_prediction.csv",
        to_s3: bool = False,
        s3_bucket: str = None,
        s3_prefix: str = None
    ) -> tuple[pd.DataFrame, str|None]:
        
        df = self._prepare_sample_dataframe()

        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        # 上传 S3（可选）
        s3_uri = None
        if to_s3:
            if not s3_bucket or not s3_prefix:
                raise ValueError("to_s3=True 时必须提供 s3_bucket/s3_prefix")
            s3 = boto3.client("s3")
            key = f"{s3_prefix}/{os.path.basename(output_path)}"
            s3.upload_file(output_path, s3_bucket, key)
            s3_uri = f"s3://{s3_bucket}/{key}"

        return df, s3_uri
    
    def get(self, key, default=None):
        if key in ('models',):
            return self.models
        if key in ('scalers',):
            return self.scalers
        if key in ('features', 'features_predict'):
            return self.features_predict
        return default

    def __getitem__(self, key):
        if key == 'models':
            return self.models
        if key == 'scalers':
            return self.scalers
        if key in ('features', 'features_predict'):
            return self.features_predict
        raise KeyError(key)
    # base class 提供的接口
    def save_model(self, path: str = "model_artifact.pkl", to_s3=False, s3_bucket=None, s3_prefix=None):
        # 1. 本地序列化
        self._save_impl(path)
        tar_path = os.path.join(os.path.dirname(path),"model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as archive:
            archive.add(path, arcname=os.path.basename(path))

        if to_s3:
            if not s3_bucket or not s3_prefix:
                raise ValueError("to_s3=True 时，需要提供 s3_bucket 和 s3_prefix")
            s3 = boto3.client("s3")
            key = f"{s3_prefix}/{os.path.basename(tar_path)}"
            s3.upload_file(tar_path, s3_bucket, key)
            return f"s3://{s3_bucket}/{key}"

        return tar_path
    
    # 需要子類別實作的function 
    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
       ...
    @abc.abstractmethod
    def _save_impl(self, path: str):
        ...

    @abc.abstractmethod
    def _load_impl(self, path: str):
        ...

    @abc.abstractmethod
    def data_preprocessing(self): ...

    @abc.abstractmethod
    def train(self): ...

    @abc.abstractmethod
    def _prepare_sample_dataframe(self) -> pd.DataFrame: ...

# ====== XGBoost Trainer ======
class XGBTrainer(BaseModelTrainer):
    def data_preprocessing(self):
        df = self.filtered_data.dropna(subset=[self.target_col] + self.features_predict)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df[self.features_predict])
        y = df[self.target_col].values
        split = int(len(df) * 0.8)
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]

    def train(self):
        self.model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42)
        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=False)
        print("XGB MAE:", np.mean(np.abs(self.model.predict(self.X_test) - self.y_test)))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[self.features_predict])
        df = df.copy()
        X = self.scaler.transform(df[self.features_predict])
        gap_pred = self.model.predict(X)
        df['OrderGapDays'] = gap_pred
        df['OrderDate'] = pd.to_datetime(df['OrderDate']).dt.date
        df['PredictedNextDate'] = df['OrderDate'] + pd.to_timedelta(df['OrderGapDays'], unit='D')
        # 按照你要的欄位輸出
        return df[['CustomerID', 'OrderDate', 'OrderGapDays', 'PredictedNextDate']]
    # 準備預測的 sample dataframe
    def _prepare_sample_dataframe(self) -> pd.DataFrame:
    # 假設你要抓每個 CustomerID 最新一筆訂單
        df_sorted = self.filtered_data.sort_values(["CustomerID", "OrderDate"])
        latest = df_sorted.groupby("CustomerID").tail(1)
        # 只留下要預測用到的欄位
        columns = ["CustomerID", "OrderDate"] + self.features_predict
        sample_df = latest[columns].copy()
        return sample_df
    # 子類別實作的儲存函數
    def _save_impl(
        self,
        path='model_artifact.pkl',
    ):
        # save metadata
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(type(self))
    def _load_impl(self, path: str):
        pass
# ====== Pytorch LSTM Trainer ======
class TorchTrainer(BaseModelTrainer):
    def __init__(self, features_predict, filtered_data, target_col, seq_len=3):
        super().__init__(features_predict, filtered_data, target_col)
        self.seq_len = seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.LSTM(input_size=len(features_predict), hidden_size=32, num_layers=1, batch_first=True).to(self.device)
        self.head = nn.Linear(32, 1).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.head.parameters()), lr=1e-3)

    def data_preprocessing(self):
        df = self.filtered_data.sort_values(["CustomerID", "OrderDate"])
        X_list, y_list = [], []
        for _, grp in df.groupby("CustomerID"):
            feats = grp[self.features_predict].values
            targets = grp[self.target_col].values
            for i in range(len(feats) - self.seq_len):
                X_list.append(feats[i:i+self.seq_len])
                y_list.append(targets[i+self.seq_len])
        X = np.stack(X_list)
        y = np.array(y_list)
        self.scaler = StandardScaler()
        N, T, F = X.shape
        X_flat = X.reshape(-1, F)
        X_scaled = self.scaler.fit_transform(X_flat).reshape(N, T, F)
        split = int(N * 0.8)
        self.train_dataset = TensorDataset(torch.tensor(X_scaled[:split], dtype=torch.float32),
                                           torch.tensor(y[:split], dtype=torch.float32))
        self.val_dataset = TensorDataset(torch.tensor(X_scaled[split:], dtype=torch.float32),
                                         torch.tensor(y[split:], dtype=torch.float32))

    def train(self, epochs=20, batch_size=64):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            self.model.train()
            tot, cnt = 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out, _ = self.model(xb)
                pred = self.head(out[:, -1, :]).squeeze()
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                tot += loss.item() * xb.size(0)
                cnt += xb.size(0)
            print(f"Epoch {epoch} train loss: {tot/cnt:.4f}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        X_list = []
        idx_list = []
        for _, grp in df.groupby('CustomerID'):
            feats = grp[self.features_predict].values
            if len(feats) < self.seq_len:
                continue
            # 只抓每個人最新一組序列來預測
            X_list.append(feats[-self.seq_len:])
            idx_list.append(grp.index[-1])
        if not X_list:
            return pd.DataFrame(columns=['CustomerID', 'OrderDate', 'OrderGapDays', 'PredictedNextDate'])
        X = np.stack(X_list)
        N, T, F = X.shape
        X_flat = X.reshape(-1, F)
        X_scaled = self.scaler.transform(X_flat).reshape(N, T, F)
        seq_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            out, _ = self.model(seq_t)
            gap_pred = self.head(out[:, -1, :]).squeeze().cpu().numpy()
        # 拼回DataFrame
        result = df.loc[idx_list, ['CustomerID', 'OrderDate']].copy()
        result['OrderGapDays'] = gap_pred
        result['OrderDate'] = pd.to_datetime(result['OrderDate']).dt.date
        result['PredictedNextDate'] = result['OrderDate'] + pd.to_timedelta(result['OrderGapDays'], unit='D')
        return result[['CustomerID', 'OrderDate', 'OrderGapDays', 'PredictedNextDate']]
    def _prepare_sample_dataframe(self) -> pd.DataFrame:
        df_sorted = self.filtered_data.sort_values(["CustomerID", "OrderDate"])
        latest = df_sorted.groupby("CustomerID").tail(self.seq_len)  # 每人抓最近 seq_len 筆
        columns = ["CustomerID", "OrderDate"] + self.features_predict
        sample_df = latest[columns].copy()
        return sample_df
    def _save_impl(
        self,
        path='model_artifact.pkl',
    ):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        

    def _load_impl(self, path: str):
        pass
# ====== Trainer Factory ======
def trainer_factory(backend, mode, features_predict, filtered_data, target_col, model_path=None, seq_len=3):
    if backend.lower() == "pytorch":
        trainer = TorchTrainer(features_predict, filtered_data, target_col, seq_len=seq_len)
    else:
        trainer = XGBTrainer(features_predict, filtered_data, target_col)
    if mode == "load" and model_path is not None:
        trainer.load(model_path)
    return trainer
