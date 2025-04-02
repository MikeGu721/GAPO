from abc import ABC, abstractmethod
import json
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error

class CustomDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "output": item["output"],
            "label": item["label"]
        }

class _BaseTrainer(ABC):
    @abstractmethod
    def train(self):
        """训练模型的抽象方法"""
        pass
    
    @abstractmethod
    def evaluate(self):
        """评估模型的抽象方法"""
        pass
    
    @abstractmethod
    def _process_batch(self, batch):
        """处理单个批次数据的抽象方法"""
        pass

class CLSTrainer(_BaseTrainer):
    def __init__(
        self,
        reward_model,
        train_data: list,
        test_data: list,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):

        '''
        format of train_data and test_data:
            {"prompt": xxx, "output": xxxx, "label": 1}
        '''
        self.reward_model = reward_model
        self.device = device
        self.reward_model.to(self.device)
        
        # 加载数据
        self.train_data = train_data
        self.test_data = test_data
        
        # 创建数据加载器
        self.train_dataset = CustomDataset(self.train_data)
        self.test_dataset = CustomDataset(self.test_data)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # 设置优化器
        self.optimizer = torch.optim.Adam(
            self.reward_model.parameters(),
            lr=learning_rate
        )
        
        self.num_epochs = num_epochs
    
    
    def _process_batch(self, batch):
        """处理单个批次的数据"""
        prompts = batch["prompt"]
        outputs = batch["output"]
        labels = torch.tensor(batch["label"]).float().to(self.device)
        
        predictions = self.reward_model(prompts, outputs)
        return predictions, labels
    
    def train(self):
        """训练模型"""
        self.reward_model.train()
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}')
            
            for batch in progress_bar:
                self.optimizer.zero_grad()
                
                predictions, labels = self._process_batch(batch)
                
                # 计算损失
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    predictions,
                    labels
                )
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
            
            # 每个epoch后评估
            test_metrics = self.evaluate()
            print(f"Epoch {epoch + 1} Test Metrics:", test_metrics)
    
    def evaluate(self):
        """评估模型"""
        self.reward_model.eval()
        all_predictions = []
        all_labels = []
        total_mse = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                predictions, labels = self._process_batch(batch)
                
                # 转换预测值为概率
                probs = torch.sigmoid(predictions)
                predicted_labels = (probs > 0.5).float()
                
                # 收集预测结果和真实标签
                all_predictions.extend(predicted_labels.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 计算MSE
                mse = mean_squared_error(
                    labels.cpu().numpy(),
                    probs.cpu().numpy()
                )
                total_mse += mse
                num_batches += 1
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # 计算各项指标
        metrics = {
            "accuracy": (all_predictions == all_labels).mean(),
            "precision": precision_score(all_labels, all_predictions, zero_division=0),
            "recall": recall_score(all_labels, all_predictions, zero_division=0),
            "f1": f1_score(all_labels, all_predictions, zero_division=0),
            "mse": total_mse / num_batches
        }
        
        return metrics

# 使用示例
if __name__ == "__main__":
    # 假设你有一个定义好的reward_model
    trainer = ModelTrainer(
        reward_model=reward_model,  # 你的reward model
        train_data_path="path/to/train.jsonl",
        test_data_path="path/to/test.jsonl",
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=10
    )
    
    # 开始训练
    trainer.train()