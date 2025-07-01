# inference_trainer.py
import pandas as pd
# 直接讀取model的predict方法
class InferenceTrainer:
    def __init__(self, trainer_instance):
        self.trainer = trainer_instance
        print("InferenceTrainer initialized with trainer instance.")
        print(f"Trainer instance type: {type(self.trainer)}")
        print(f"Trainer instance attributes: {dir(self.trainer)}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.trainer.predict(df)