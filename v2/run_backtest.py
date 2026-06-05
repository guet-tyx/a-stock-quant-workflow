#!/usr/bin/env python3
"""
A-Share Multi-Factor Strategy - Walk-Forward Backtest (v2)

Usage:
    python run_backtest.py

Requires TUSHARE_TOKEN environment variable.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Add modules path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.factor_engine import FactorEngine
from modules.backtest_engine import WalkForwardBacktest

OUTPUT_DIR = os.path.expanduser('~/quant_strategies/individual_stock_strategy/v2/output')
DATA_DIR = os.path.expanduser('~/quant_strategies/individual_stock_strategy/data')

def main():
    print('=' * 60)
    print('A-Share Multi-Factor Strategy - Walk-Forward Backtest (v2)')
    print('=' * 60)
    
    # Step 1: Load data
    print('\n[Step 1] Loading data...')
    engine = FactorEngine()
    engine.load_data(f'{DATA_DIR}/daily_data_fixed.csv')
    
    # Step 2: Compute factors
    print('\n[Step 2] Computing all factors...')
    engine.compute_all_factors()
    
    # Step 3: Compute IC for all factors
    print('\n[Step 3] Computing factor IC...')
    engine.factor_df['ym'] = engine.factor_df['trade_date'].dt.to_period('M')
    month_end = engine.factor_df.groupby(['ts_code', 'ym'])['trade_date'].transform('max')
    monthly = engine.factor_df[engine.factor_df['trade_date'] == month_end].copy()
    
    ic_df = engine.compute_all_ic(monthly)
    ic_df.to_csv(f'{OUTPUT_DIR}/factor_ic_v2.csv', index=False)
    print(f'Saved IC analysis ({len(ic_df)} factors)')
    
    # Step 4: Run walk-forward backtest
    print('\n[Step 4] Running Walk-Forward backtest...')
    
    configs = [
        {'name': 'combo_1', 'top_n': 20, 'max_turnover': 0.3, 'train_window': 24, 'retrain_freq': 6},
        {'name': 'combo_2', 'top_n': 30, 'max_turnover': 0.3, 'train_window': 24, 'retrain_freq': 6},
        {'name': 'combo_3', 'top_n': 20, 'max_turnover': 0.5, 'train_window': 24, 'retrain_freq': 6},
        {'name': 'combo_4', 'top_n': 30, 'max_turnover': 0.3, 'train_window': 18, 'retrain_freq': 3},
        {'name': 'combo_5', 'top_n': 20, 'max_turnover': 0.3, 'train_window': 18, 'retrain_freq': 3},
    ]
    
    for cfg in configs:
        print(f'\n  Running {cfg["name"]}: top_n={cfg["top_n"]}, turnover={cfg["max_turnover"]}, train={cfg["train_window"]}m, retrain={cfg["retrain_freq"]}m')
        bt = WalkForwardBacktest(engine.factor_df, config=cfg)
        result = bt.run()
        if result:
            s = result['summary']
            print(f'  → AnnRet: {s["annual_return"]*100:.2f}%, Sharpe: {s["sharpe_ratio"]:.2f}, MDD: {s["max_drawdown"]*100:.2f}%')
    
    print('\n✅ v2 backtest complete!')

if __name__ == '__main__':
    main()
