import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

class MLModel:
    def __init__(self, model_type='rf'):
        """
        初始化机器学习模型
        model_type: 'rf'(随机森林), 'gbm'(梯度提升), 'xgb'(XGBoost)
        """
        self.model_type = model_type
        self.model = None
        
        if model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gbm':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgb':
            self.model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("不支持的模型类型，请选择'rf'、'gbm'或'xgb'")
    
    def train(self, X, y, tune_hyperparams=False):
        """训练模型"""
        # 分割训练集和测试集，使用时间序列分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        if tune_hyperparams:
            # 设置超参数网格搜索
            if self.model_type == 'rf':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'gbm':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7]
                }
            elif self.model_type == 'xgb':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            
            # 使用时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)
            
            # 网格搜索
            grid_search = GridSearchCV(
                self.model, param_grid, cv=tscv, 
                scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # 使用最佳参数更新模型
            self.model = grid_search.best_estimator_
            print(f"最佳参数: {grid_search.best_params_}")
        else:
            self.model.fit(X_train, y_train)
        
        # 评估模型
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"训练集 MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"测试集 MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        
        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = X.columns
            indices = np.argsort(importances)[::-1]
            
            print("\n特征重要性:")
            for i in range(min(10, len(feature_names))):
                print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return {
            'model': self.model,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'train_pred': train_pred, 'test_pred': test_pred
        }
    
    def predict(self, X):
        """使用训练好的模型进行预测"""
        if self.model is None:
            raise Exception("请先训练模型")
        
        return self.model.predict(X)
    
    def plot_feature_importance(self, X, top_n=10):
        """绘制特征重要性"""
        if not hasattr(self.model, 'feature_importances_'):
            print("当前模型不支持特征重要性")
            return
        
        importances = self.model.feature_importances_
        feature_names = X.columns
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('特征重要性')
        plt.bar(range(min(top_n, len(feature_names))), 
                importances[indices[:top_n]], 
                align='center')
        plt.xticks(range(min(top_n, len(feature_names))), 
                  [feature_names[i] for i in indices[:top_n]], 
                  rotation=90)
        plt.tight_layout()
        plt.show()
    
    def plot_prediction(self, y_true, y_pred, title='预测vs实际'):
        """绘制预测结果对比图"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.values, label='实际值')
        plt.plot(y_pred, label='预测值', alpha=0.7)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show() 