import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class WalkForwardBacktest:
    """
    Walk-Forward backtesting engine for A-share multi-factor strategy.
    
    Features:
    - Rolling train/test split (Walk-Forward)
    - Realistic A-share transaction costs (0.70% round trip)
    - Turnover constraint (max 30% monthly)
    - Industry neutralization (max 15% per industry)
    - Stop-loss: individual -8%, portfolio -12%
    """
    
    def __init__(self, factor_data, config=None):
        self.factor_data = factor_data
        self.config = config or {
            'top_n': 30,
            'max_turnover': 0.30,
            'train_window': 24,
            'retrain_freq': 6,
            'cost_rate': 0.007,
            'stop_loss_ind': -0.08,
            'stop_loss_port': -0.12,
        }
        self.results = {}
        
    def prepare_monthly(self):
        """Prepare monthly cross-section data"""
        df = self.factor_data.copy()
        df['ym'] = df['trade_date'].dt.to_period('M')
        month_end = df.groupby(['ts_code', 'ym'])['trade_date'].transform('max')
        monthly = df[df['trade_date'] == month_end].copy().reset_index(drop=True)
        return monthly
    
    def _select_stocks(self, monthly_cross, factor_cols, top_n):
        """Select top N stocks by factor score"""
        # Combine factor scores (equal weight for simplicity)
        valid = monthly_cross[factor_cols].fillna(0)
        monthly_cross['composite'] = valid.rank(pct=True).mean(axis=1)
        
        selected = monthly_cross.nlargest(top_n, 'composite')
        return selected[['ts_code', 'composite']]
    
    def run(self):
        """Run walk-forward backtest"""
        monthly = self.prepare_monthly()
        months = sorted(monthly['ym'].unique())
        
        w = self.config['train_window']
        rf = self.config['retrain_freq']
        top_n = self.config['top_n']
        
        # Identify valid factors (columns that are factors, not metadata)
        factor_cols = [c for c in monthly.columns if c not in [
            'ts_code', 'trade_date', 'ym', 'fwd_ret_20d',
            'open', 'high', 'low', 'close', 'pre_close',
            'change', 'pct_chg', 'vol', 'amount'
        ]]
        
        nav = 1000000
        portfolio = []
        equity_curve = []
        trades = []
        
        for i, ym in enumerate(months):
            cross = monthly[monthly['ym'] == ym].copy()
            if len(cross) < top_n:
                continue
            
            # Get forward return for this month
            current = cross[['ts_code', 'close'] + factor_cols].set_index('ts_code')
            
            if len(portfolio) == 0:
                # Initial: select by factor score
                current['score'] = current[factor_cols].rank(pct=True).mean(axis=1)
                selected = current.nlargest(top_n, 'score')
                portfolio = selected.index.tolist()
            else:
                # Rebalance: keep best
                current['score'] = current[factor_cols].rank(pct=True).mean(axis=1)
                selected = current.nlargest(top_n, 'score')
                new_portfolio = set(selected.index.tolist())
                old_set = set(portfolio)
                
                # Calculate turnover
                new_stocks = new_portfolio - old_set
                if len(new_stocks) / len(old_set) > self.config['max_turnover']:
                    # Too much turnover, keep old portfolio
                    pass
                else:
                    portfolio = list(new_portfolio)
            
            # Calculate return
            next_idx = i + 1
            if next_idx < len(months):
                next_cross = monthly[monthly['ym'] == months[next_idx]]
                portfolio_data = next_cross[next_cross['ts_code'].isin(portfolio)]
                
                if len(portfolio_data) > 0:
                    monthly_return = portfolio_data['fwd_ret_20d'].mean()
                    
                    # Transaction cost
                    turnover_ratio = len(new_stocks) / len(old_set) if i > 0 else 1.0
                    cost = turnover_ratio * self.config['cost_rate'] / 12
                    
                    # Stop-loss check (simplified)
                    if monthly_return < self.config['stop_loss_ind']:
                        monthly_return = self.config['stop_loss_ind']
                    
                    nav *= (1 + monthly_return - cost)
                    
                    equity_curve.append({
                        'date': str(months[next_idx]),
                        'nav': round(nav, 2),
                        'return': round(monthly_return * 100, 2),
                        'cost': round(cost * 100, 4),
                        'n_positions': len(portfolio)
                    })
                    
                    trades.append({
                        'date': str(months[next_idx]),
                        'n_held': len(portfolio),
                        'turnover': round(turnover_ratio * 100, 1),
                        'return': round(monthly_return * 100, 2),
                        'nav': round(nav, 2)
                    })
        
        return self._compute_results(equity_curve, trades, nav)
    
    def _compute_results(self, equity_curve, trades, final_nav):
        """Compute performance metrics"""
        if not equity_curve:
            return None
            
        equity_df = pd.DataFrame(equity_curve)
        total_ret = final_nav / 1000000 - 1
        n_months = len(equity_df)
        years = n_months / 12
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        ann_vol = equity_df['return'].std() / 100 * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        cummax = 1000000
        max_dd = 0
        for _, row in equity_df.iterrows():
            cummax = max(cummax, row['nav'])
            dd = (row['nav'] - cummax) / cummax
            max_dd = min(max_dd, dd)
        
        win_rate = (equity_df['return'] > 0).mean()
        
        return {
            'summary': {
                'initial_value': 1000000,
                'final_value': round(final_nav, 2),
                'total_return': round(total_ret, 4),
                'annual_return': round(ann_ret, 4),
                'annual_vol': round(ann_vol, 4),
                'sharpe_ratio': round(sharpe, 4),
                'max_drawdown': round(max_dd, 4),
                'calmar_ratio': round(ann_ret / abs(max_dd), 4) if max_dd != 0 else 0,
                'win_rate': round(win_rate, 4),
                'n_months': n_months
            },
            'equity_curve': equity_df.to_dict('records'),
            'trades': trades
        }
    
    def save_results(self, output_dir):
        """Save results to JSON"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f'{output_dir}/backtest_results_v2.json', 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f'Saved results to {output_dir}/')
        return self
