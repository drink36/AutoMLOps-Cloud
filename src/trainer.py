# common 前處理
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class DataTrainer:
    def __init__(self, file_paths, sequence_length=30):
        self.file_paths = file_paths
        self.data = None
        self.filtered_data = None
        self.customer_features = None
        self.encoder = None
        self.lstm_features = None
        self.seq_customer_ids = None
        self.cluster_test_data = {}
        self.sequence_length = sequence_length
        # 特徵列表
        self.features_predict = [
            "OrderGapDays",        # 前一筆到這筆的天數差
            "RollingAvgGap",       # 最近3筆gap平均
            "OrderCount",          # 累積下單次數
            "TotalPrice",          # 單日消費總金額
            "Quantity",            # 單日購買總數量
            "CumulativeTotalPrice",# 累積消費總金額
            "year",
            "weekday_sin",
            "weekday_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
            "quarter_sin",
            "quarter_cos",
        ]
        # 時間序列特徵
        self.time_series_features = [
            "year",
            "weekday_sin",
            "weekday_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
            "quarter_sin",
            "quarter_cos",
        ]
    def load_data(self):
        data = pd.read_csv(self.file_paths["all"])
        data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
        # sum total price and quantity by CustomerID and InvoiceDate 
        data = data.groupby(['CustomerID', data['InvoiceDate'].dt.date]).agg({'TotalPrice': 'sum', 'Quantity': 'sum'}).reset_index()
        # 把剛剛groupby後的date欄位重新命名
        data = data.rename(columns={'InvoiceDate': 'OrderDate'})
        # 再把OrderDate轉回datetime64（方便做dt操作）
        data['OrderDate'] = pd.to_datetime(data['OrderDate'])
        # only keep rows where the TotalPrice is greater than 0
        data = data[data['TotalPrice'] > 0]
        # # left the customer who has more than 10 orders
        # data = data.groupby('CustomerID').filter(lambda x: len(x) > 10)
        # 時序排序
        data = data.sort_values(['CustomerID', 'OrderDate'])

        # 前一筆下單日
        data['PrevOrderDate'] = data.groupby('CustomerID')['OrderDate'].shift(1)
        data['OrderGapDays'] = (data['OrderDate'] - data['PrevOrderDate']).dt.days
        data = data[data['OrderGapDays'] > 0].copy()
        data['RollingAvgGap'] = data.groupby('CustomerID')['OrderGapDays'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        # 累積下單天數
        data['OrderCount'] = data.groupby('CustomerID').cumcount() + 1
        # 累積下單金額
        data['CumulativeTotalPrice'] = data.groupby('CustomerID')['TotalPrice'].cumsum()
        # sin cos feature engineering
        data["month"] = data["OrderDate"].dt.month
        data["weekday"] = data["OrderDate"].dt.weekday
        data["year"] = data["OrderDate"].dt.year
        data["day"] = data["OrderDate"].dt.day
        data["quarter"] = data["OrderDate"].dt.quarter
        data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
        data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)
        data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
        data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['quarter_sin'] = np.sin(2 * np.pi * data['quarter'] / 4)
        data['quarter_cos'] = np.cos(2 * np.pi * data['quarter'] / 4)
        # delete 不需要的欄位
        data = data.drop(columns=['month', 'weekday', 'day', 'quarter'])
        
        self.data = data.copy()
    
    def filter_outliers(self):
        customer_features = self.data.groupby(["CustomerID"])[self.features_predict].mean().fillna(0)

        iso_forest = IsolationForest(contamination=0.01, random_state=42)
        outlier_flags = iso_forest.fit_predict(customer_features)

        normal_customers = customer_features.index[outlier_flags == 1].tolist()

        self.filtered_data = self.data[self.data["CustomerID"].isin(normal_customers)].copy()
    def prepare_processed_data(self):
        """
        將資料轉換為適合訓練的格式
        """
        data = self.filtered_data.copy()
        data = data.sort_values(by=["CustomerID", "OrderDate"])
        return data, self.features_predict, self.time_series_features
    
    def filter_pipeline(self):
        self.load_data()
        self.filter_outliers()