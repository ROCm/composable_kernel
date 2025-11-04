#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
ML-Based Auto-Tuner using XGBoost

Train an XGBoost model on tile_engine performance data to predict
the best kernel configuration for any given problem size.

Features:
- Learn from historical tile_engine benchmarks
- Predict performance for unseen configurations
- Recommend optimal kernel for any problem size
- Feature engineering for GEMM characteristics
- Model persistence and versioning
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

log = logging.getLogger(__name__)

# Optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    log.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    log.warning("Pandas not available. Install with: pip install pandas")


# ============================================================================
# Performance Data Structures
# ============================================================================

@dataclass
class KernelPerformanceData:
    """Performance data for a single kernel configuration"""
    # Problem characteristics
    M: int
    N: int
    K: int
    batch_size: int = 1
    
    # Kernel configuration
    tile_m: int = 0
    tile_n: int = 0
    tile_k: int = 0
    warp_m: int = 0
    warp_n: int = 0
    warp_k: int = 0
    warp_tile_m: int = 0
    warp_tile_n: int = 0
    warp_tile_k: int = 0
    block_size: int = 256
    
    # Kernel traits
    pipeline: str = "compv4"
    epilogue: str = "cshuffle"
    scheduler: str = "intrawave"
    persistent: bool = False
    
    # Data types
    dtype_a: str = "fp16"
    dtype_b: str = "fp16"
    dtype_c: str = "fp16"
    
    # Performance metrics
    execution_time_ms: float = 0.0
    gflops: float = 0.0
    memory_bandwidth_gb_s: float = 0.0
    occupancy: float = 0.0
    
    # Hardware info
    gpu_arch: str = "gfx942"
    num_cus: int = 304
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def compute_gflops(self):
        """Compute GFLOPS from execution time"""
        if self.execution_time_ms > 0:
            flops = 2.0 * self.M * self.N * self.K * self.batch_size
            self.gflops = flops / (self.execution_time_ms * 1e6)


# ============================================================================
# Feature Engineering
# ============================================================================

class FeatureEngineer:
    """Extract and engineer features for ML model"""
    
    @staticmethod
    def extract_features(data: KernelPerformanceData) -> Dict[str, float]:
        """
        Extract features from performance data
        
        Returns dictionary of features suitable for ML model
        """
        features = {}
        
        # Problem size features
        features['M'] = float(data.M)
        features['N'] = float(data.N)
        features['K'] = float(data.K)
        features['batch_size'] = float(data.batch_size)
        
        # Derived problem features
        features['problem_size'] = float(data.M * data.N * data.K)
        features['M_div_N'] = float(data.M) / max(float(data.N), 1.0)
        features['N_div_K'] = float(data.N) / max(float(data.K), 1.0)
        features['M_div_K'] = float(data.M) / max(float(data.K), 1.0)
        features['max_dim'] = float(max(data.M, data.N, data.K))
        features['min_dim'] = float(min(data.M, data.N, data.K))
        features['dim_ratio'] = features['max_dim'] / max(features['min_dim'], 1.0)
        
        # Tile configuration features
        features['tile_m'] = float(data.tile_m)
        features['tile_n'] = float(data.tile_n)
        features['tile_k'] = float(data.tile_k)
        features['tile_size'] = float(data.tile_m * data.tile_n * data.tile_k)
        
        # Warp configuration features
        features['warp_m'] = float(data.warp_m)
        features['warp_n'] = float(data.warp_n)
        features['warp_k'] = float(data.warp_k)
        features['warps_per_block'] = float(data.warp_m * data.warp_n * data.warp_k)
        
        # Warp tile features
        features['warp_tile_m'] = float(data.warp_tile_m)
        features['warp_tile_n'] = float(data.warp_tile_n)
        features['warp_tile_k'] = float(data.warp_tile_k)
        features['warp_tile_size'] = float(data.warp_tile_m * data.warp_tile_n * data.warp_tile_k)
        
        # Block features
        features['block_size'] = float(data.block_size)
        
        # Tile coverage features (how many tiles needed)
        features['num_tiles_m'] = float(data.M) / max(float(data.tile_m), 1.0)
        features['num_tiles_n'] = float(data.N) / max(float(data.tile_n), 1.0)
        features['num_tiles_k'] = float(data.K) / max(float(data.tile_k), 1.0)
        features['total_tiles'] = features['num_tiles_m'] * features['num_tiles_n']
        
        # Tile efficiency (how well tiles fit problem)
        features['tile_efficiency_m'] = 1.0 if data.M % data.tile_m == 0 else float(data.M % data.tile_m) / float(data.tile_m)
        features['tile_efficiency_n'] = 1.0 if data.N % data.tile_n == 0 else float(data.N % data.tile_n) / float(data.tile_n)
        features['tile_efficiency_k'] = 1.0 if data.K % data.tile_k == 0 else float(data.K % data.tile_k) / float(data.tile_k)
        
        # Arithmetic intensity
        flops = 2.0 * data.M * data.N * data.K
        memory_bytes = (data.M * data.K + data.K * data.N + data.M * data.N) * 2  # fp16
        features['arithmetic_intensity'] = flops / max(memory_bytes, 1.0)
        
        # Categorical features (one-hot encoded)
        features['pipeline_compv3'] = 1.0 if data.pipeline == "compv3" else 0.0
        features['pipeline_compv4'] = 1.0 if data.pipeline == "compv4" else 0.0
        features['pipeline_mem'] = 1.0 if data.pipeline == "mem" else 0.0
        
        features['epilogue_cshuffle'] = 1.0 if data.epilogue == "cshuffle" else 0.0
        features['epilogue_default'] = 1.0 if data.epilogue == "default" else 0.0
        
        features['scheduler_intrawave'] = 1.0 if data.scheduler == "intrawave" else 0.0
        features['scheduler_interwave'] = 1.0 if data.scheduler == "interwave" else 0.0
        
        features['persistent'] = 1.0 if data.persistent else 0.0
        
        # Datatype features
        features['dtype_fp16'] = 1.0 if data.dtype_a == "fp16" else 0.0
        features['dtype_bf16'] = 1.0 if data.dtype_a == "bf16" else 0.0
        features['dtype_fp32'] = 1.0 if data.dtype_a == "fp32" else 0.0
        features['dtype_int8'] = 1.0 if data.dtype_a == "int8" else 0.0
        
        # Hardware features
        features['num_cus'] = float(data.num_cus)
        
        return features
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of all feature names"""
        # Create dummy data to extract feature names
        dummy = KernelPerformanceData(
            M=128, N=128, K=128,
            tile_m=128, tile_n=128, tile_k=32,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16
        )
        features = FeatureEngineer.extract_features(dummy)
        return list(features.keys())


# ============================================================================
# Data Loader
# ============================================================================

class TileEngineDataLoader:
    """Load performance data from tile_engine benchmarks"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
    
    def load_from_json(self, json_path: Path) -> List[KernelPerformanceData]:
        """
        Load performance data from JSON file
        
        Expected format:
        {
            "benchmarks": [
                {
                    "problem": {"M": 128, "N": 128, "K": 128},
                    "config": {"tile_m": 128, "tile_n": 128, "tile_k": 32, ...},
                    "performance": {"execution_time_ms": 0.5, "gflops": 100.0, ...}
                },
                ...
            ]
        }
        """
        if not json_path.exists():
            log.error(f"Data file not found: {json_path}")
            return []
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        performance_data = []
        
        for benchmark in data.get('benchmarks', []):
            try:
                problem = benchmark.get('problem', {})
                config = benchmark.get('config', {})
                perf = benchmark.get('performance', {})
                
                entry = KernelPerformanceData(
                    M=problem.get('M', 0),
                    N=problem.get('N', 0),
                    K=problem.get('K', 0),
                    batch_size=problem.get('batch_size', 1),
                    
                    tile_m=config.get('tile_m', 0),
                    tile_n=config.get('tile_n', 0),
                    tile_k=config.get('tile_k', 0),
                    warp_m=config.get('warp_m', 0),
                    warp_n=config.get('warp_n', 0),
                    warp_k=config.get('warp_k', 0),
                    warp_tile_m=config.get('warp_tile_m', 0),
                    warp_tile_n=config.get('warp_tile_n', 0),
                    warp_tile_k=config.get('warp_tile_k', 0),
                    block_size=config.get('block_size', 256),
                    
                    pipeline=config.get('pipeline', 'compv4'),
                    epilogue=config.get('epilogue', 'cshuffle'),
                    scheduler=config.get('scheduler', 'intrawave'),
                    persistent=config.get('persistent', False),
                    
                    dtype_a=config.get('dtype_a', 'fp16'),
                    dtype_b=config.get('dtype_b', 'fp16'),
                    dtype_c=config.get('dtype_c', 'fp16'),
                    
                    execution_time_ms=perf.get('execution_time_ms', 0.0),
                    gflops=perf.get('gflops', 0.0),
                    memory_bandwidth_gb_s=perf.get('memory_bandwidth_gb_s', 0.0),
                    occupancy=perf.get('occupancy', 0.0),
                    
                    gpu_arch=config.get('gpu_arch', 'gfx942'),
                    num_cus=config.get('num_cus', 304),
                )
                
                # Compute GFLOPS if not provided
                if entry.gflops == 0.0 and entry.execution_time_ms > 0.0:
                    entry.compute_gflops()
                
                performance_data.append(entry)
                
            except Exception as e:
                log.warning(f"Failed to parse benchmark entry: {e}")
                continue
        
        log.info(f"Loaded {len(performance_data)} performance entries from {json_path}")
        return performance_data
    
    def load_from_csv(self, csv_path: Path) -> List[KernelPerformanceData]:
        """Load performance data from CSV file"""
        if not HAS_PANDAS:
            log.error("Pandas required for CSV loading")
            return []
        
        if not csv_path.exists():
            log.error(f"Data file not found: {csv_path}")
            return []
        
        df = pd.read_csv(csv_path)
        
        performance_data = []
        for _, row in df.iterrows():
            try:
                entry = KernelPerformanceData(**row.to_dict())
                if entry.gflops == 0.0 and entry.execution_time_ms > 0.0:
                    entry.compute_gflops()
                performance_data.append(entry)
            except Exception as e:
                log.warning(f"Failed to parse row: {e}")
                continue
        
        log.info(f"Loaded {len(performance_data)} performance entries from {csv_path}")
        return performance_data
    
    def scan_directory(self) -> List[KernelPerformanceData]:
        """Scan directory for all benchmark files"""
        all_data = []
        
        # Load JSON files
        for json_file in self.data_dir.glob("**/*.json"):
            data = self.load_from_json(json_file)
            all_data.extend(data)
        
        # Load CSV files
        if HAS_PANDAS:
            for csv_file in self.data_dir.glob("**/*.csv"):
                data = self.load_from_csv(csv_file)
                all_data.extend(data)
        
        log.info(f"Total performance entries loaded: {len(all_data)}")
        return all_data


# ============================================================================
# XGBoost Model
# ============================================================================

class XGBoostAutoTuner:
    """XGBoost-based auto-tuner for GEMM kernels"""
    
    def __init__(self, model_dir: Path = Path("./models")):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: List[str] = []
        self.scaler_params: Optional[Dict] = None
        
        if not HAS_XGBOOST:
            raise ImportError("XGBoost required. Install with: pip install xgboost")
    
    def train(
        self,
        training_data: List[KernelPerformanceData],
        target_metric: str = "gflops",
        test_split: float = 0.2,
        **xgb_params
    ) -> Dict[str, float]:
        """
        Train XGBoost model on performance data
        
        Args:
            training_data: List of performance data
            target_metric: Metric to predict ('gflops', 'execution_time_ms', etc.)
            test_split: Fraction of data for testing
            **xgb_params: Additional XGBoost parameters
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not training_data:
            raise ValueError("No training data provided")
        
        log.info(f"Training XGBoost model on {len(training_data)} samples")
        
        # Extract features and targets
        X = []
        y = []
        
        for data in training_data:
            features = FeatureEngineer.extract_features(data)
            X.append(list(features.values()))
            y.append(getattr(data, target_metric))
        
        X = np.array(X)
        y = np.array(y)
        
        self.feature_names = list(FeatureEngineer.extract_features(training_data[0]).keys())
        
        # Split data
        n_test = int(len(X) * test_split)
        indices = np.random.permutation(len(X))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Normalize features
        self.scaler_params = {
            'mean': X_train.mean(axis=0),
            'std': X_train.std(axis=0) + 1e-8
        }
        
        X_train = (X_train - self.scaler_params['mean']) / self.scaler_params['std']
        X_test = (X_test - self.scaler_params['mean']) / self.scaler_params['std']
        
        # Default XGBoost parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42,
        }
        default_params.update(xgb_params)
        
        # Train model
        self.model = xgb.XGBRegressor(**default_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mse': float(np.mean((y_train - train_pred) ** 2)),
            'test_mse': float(np.mean((y_test - test_pred) ** 2)),
            'train_mae': float(np.mean(np.abs(y_train - train_pred))),
            'test_mae': float(np.mean(np.abs(y_test - test_pred))),
            'train_r2': float(1 - np.sum((y_train - train_pred) ** 2) / np.sum((y_train - y_train.mean()) ** 2)),
            'test_r2': float(1 - np.sum((y_test - test_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)),
        }
        
        log.info(f"Training complete. Test R²: {metrics['test_r2']:.4f}, Test MAE: {metrics['test_mae']:.4f}")
        
        return metrics
    
    def predict(self, config: KernelPerformanceData) -> float:
        """Predict performance for a configuration"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        features = FeatureEngineer.extract_features(config)
        X = np.array([list(features.values())])
        
        # Normalize
        X = (X - self.scaler_params['mean']) / self.scaler_params['std']
        
        prediction = self.model.predict(X)[0]
        return float(prediction)
    
    def recommend_best_config(
        self,
        problem_size: Tuple[int, int, int],
        candidate_configs: List[KernelPerformanceData],
        batch_size: int = 1
    ) -> Tuple[KernelPerformanceData, float]:
        """
        Recommend best configuration for problem size
        
        Args:
            problem_size: (M, N, K)
            candidate_configs: List of candidate configurations
            batch_size: Batch size
        
        Returns:
            (best_config, predicted_performance)
        """
        M, N, K = problem_size
        
        best_config = None
        best_performance = -float('inf')
        
        for config in candidate_configs:
            # Update problem size
            test_config = KernelPerformanceData(**config.to_dict())
            test_config.M = M
            test_config.N = N
            test_config.K = K
            test_config.batch_size = batch_size
            
            # Predict performance
            predicted_perf = self.predict(test_config)
            
            if predicted_perf > best_performance:
                best_performance = predicted_perf
                best_config = test_config
        
        return best_config, best_performance
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def save_model(self, model_path: Path):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'scaler_params': self.scaler_params,
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        log.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Path):
        """Load model from disk"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.scaler_params = model_data['scaler_params']
        
        log.info(f"Model loaded from {model_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ML-based auto-tuner for GEMM kernels')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data-dir', type=Path, required=True,
                             help='Directory containing benchmark data')
    train_parser.add_argument('--output', type=Path, default=Path('./models/autotuner.pkl'),
                             help='Output model path')
    train_parser.add_argument('--target', type=str, default='gflops',
                             choices=['gflops', 'execution_time_ms'],
                             help='Target metric to predict')
    train_parser.add_argument('--test-split', type=float, default=0.2,
                             help='Test split fraction')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict performance')
    predict_parser.add_argument('--model', type=Path, required=True,
                               help='Model path')
    predict_parser.add_argument('--problem-size', nargs=3, type=int, required=True,
                               metavar=('M', 'N', 'K'))
    predict_parser.add_argument('--config', type=Path, required=True,
                               help='Kernel configuration JSON')
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Recommend best config')
    recommend_parser.add_argument('--model', type=Path, required=True,
                                 help='Model path')
    recommend_parser.add_argument('--problem-size', nargs=3, type=int, required=True,
                                 metavar=('M', 'N', 'K'))
    recommend_parser.add_argument('--candidates', type=Path, required=True,
                                 help='Candidate configurations JSON')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Load data
        loader = TileEngineDataLoader(args.data_dir)
        training_data = loader.scan_directory()
        
        if not training_data:
            print("No training data found!")
            return 1
        
        # Train model
        tuner = XGBoostAutoTuner()
        metrics = tuner.train(training_data, target_metric=args.target, test_split=args.test_split)
        
        # Print metrics
        print("\nTraining Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Print feature importance
        print("\nTop 10 Important Features:")
        importance = tuner.get_feature_importance()
        for i, (feat, imp) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10], 1):
            print(f"  {i}. {feat}: {imp:.4f}")
        
        # Save model
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tuner.save_model(args.output)
        print(f"\nModel saved to {args.output}")
    
    elif args.command == 'predict':
        # Load model
        tuner = XGBoostAutoTuner()
        tuner.load_model(args.model)
        
        # Load config
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        
        M, N, K = args.problem_size
        config_dict.update({'M': M, 'N': N, 'K': K})
        
        config = KernelPerformanceData(**config_dict)
        
        # Predict
        predicted = tuner.predict(config)
        print(f"\nPredicted performance: {predicted:.2f} GFLOPS")
    
    elif args.command == 'recommend':
        # Load model
        tuner = XGBoostAutoTuner()
        tuner.load_model(args.model)
        
        # Load candidates
        with open(args.candidates, 'r') as f:
            candidates_data = json.load(f)
        
        candidates = [KernelPerformanceData(**c) for c in candidates_data]
        
        # Recommend
        M, N, K = args.problem_size
        best_config, best_perf = tuner.recommend_best_config((M, N, K), candidates)
        
        print(f"\nBest configuration for problem size ({M}, {N}, {K}):")
        print(f"  Tile: {best_config.tile_m}x{best_config.tile_n}x{best_config.tile_k}")
        print(f"  Warp: {best_config.warp_m}x{best_config.warp_n}x{best_config.warp_k}")
        print(f"  Warp Tile: {best_config.warp_tile_m}x{best_config.warp_tile_n}x{best_config.warp_tile_k}")
        print(f"  Pipeline: {best_config.pipeline}")
        print(f"  Predicted performance: {best_perf:.2f} GFLOPS")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

