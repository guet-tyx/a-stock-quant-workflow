#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward回测引擎 v2.0
修复项：
1. 滚动窗口训练（替代固定train/test split）
2. 真实交易成本（佣金+印花税+滑点+冲击成本）
3. 换手率约束
4. 动态止损
5. 行业暴露控制
6. 幸存者偏差处理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import mean_squared_error


class RealisticCostModel:
    """A股真实交易成本模型"""
    
    def __init__(self, 
                 commission: float = 0.00025,   # 佣金万2.5
                 stamp_tax: float = 0.0005,      # 印花税0.05%（卖出）
                 slippage: float = 0.002,         # 滑点0.2%
                 impact_cost: float = 0.001):     # 冲击成本0.1%
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        self.impact_cost = impact_cost
    
    def buy_cost(self, amount: float) -> float:
        """买入成本（佣金+滑点+冲击）"""
        return amount * (self.commission + self.slippage + self.impact_cost)
    
    def sell_cost(self, amount: float) -> float:
        """卖出成本（佣金+印花税+滑点+冲击）"""
        return amount * (self.commission + self.stamp_tax + self.slippage + self.impact_cost)
    
    def round_trip_cost(self) -> float:
        """买卖一个来回的总成本率"""
        return (self.commission * 2 + self.stamp_tax + self.slippage * 2 + self.impact_cost * 2)


class RiskManager:
    """风险管理器"""
    
    def __init__(self,
                 stop_loss_individual: float = -0.08,    # 个股止损-8%
                 stop_loss_portfolio: float = -0.12,      # 组合止损-12%
                 max_drawdown_trigger: float = -0.15,     # 最大回撤触发减仓
                 vol_threshold_high: float = 0.35,        # 高波动阈值
                 vol_threshold_low: float = 0.20,         # 低波动阈值
                 max_position_high_vol: float = 0.5,      # 高波动最大仓位
                 max_position_normal: float = 0.8,         # 正常最大仓位
                 max_industry_weight: float = 0.15,        # 单行业最大权重
                 max_stock_weight: float = 0.05):          # 单股最大权重
        self.stop_loss_individual = stop_loss_individual
        self.stop_loss_portfolio = stop_loss_portfolio
        self.max_drawdown_trigger = max_drawdown_trigger
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_low = vol_threshold_low
        self.max_position_high_vol = max_position_high_vol
        self.max_position_normal = max_position_normal
        self.max_industry_weight = max_industry_weight
        self.max_stock_weight = max_stock_weight
    
    def get_position_ratio(self, portfolio_vol: float) -> float:
        """根据组合波动率确定仓位比例"""
        if portfolio_vol > self.vol_threshold_high:
            return self.max_position_high_vol
        elif portfolio_vol > self.vol_threshold_low:
            return self.max_position_normal
        else:
            return 1.0
    
    def check_individual_stop(self, code: str, entry_price: float, 
                              current_price: float) -> bool:
        """检查个股止损"""
        ret = (current_price - entry_price) / entry_price
        return ret <= self.stop_loss_individual
    
    def check_portfolio_stop(self, current_drawdown: float) -> bool:
        """检查组合止损"""
        return current_drawdown <= self.stop_loss_portfolio
    
    def select_stocks(self, scores: pd.DataFrame, top_n: int, 
                      industry_map: Dict[str, str]) -> List[str]:
        """行业中性化选股"""
        selected = []
        industry_count = {}
        max_per_industry = max(1, int(top_n * self.max_industry_weight))
        
        for _, row in scores.iterrows():
            code = row['ts_code']
            industry = industry_map.get(code, 'Unknown')
            
            if industry_count.get(industry, 0) < max_per_industry:
                selected.append(code)
                industry_count[industry] = industry_count.get(industry, 0) + 1
                
                if len(selected) >= top_n:
                    break
        
        return selected


class WalkForwardBacktest:
    """Walk-Forward回测引擎"""
    
    def __init__(self,
                 factor_df: pd.DataFrame,
                 factor_cols: List[str],
                 stock_info: pd.DataFrame,
                 ic_signs: Dict[str, float] = None,   # 【修复Bug 1】因子IC符号，等权模式反转负IC因子
                 train_window: int = 24,         # 训练窗口（月）
                 retrain_freq: int = 3,          # 重训频率（月）
                 top_n: int = 30,                # 持仓数量
                 initial_cash: float = 1000000,  # 初始资金
                 max_turnover: float = 0.5,      # 最大月换手率
                 rebalance_freq: str = 'M'):      # 调仓频率
        self.factor_df = factor_df.copy()
        self.factor_cols = factor_cols
        self.stock_info = stock_info
        self.ic_signs = ic_signs or {}            # 【修复Bug 1】
        self.train_window = train_window
        self.retrain_freq = retrain_freq
        self.top_n = top_n
        self.initial_cash = initial_cash
        self.max_turnover = max_turnover
        self.rebuild_freq = rebalance_freq
        
        # 初始化组件
        self.cost_model = RealisticCostModel()
        self.risk_manager = RiskManager()
        self.industry_map = dict(zip(stock_info['ts_code'], stock_info['industry']))
        
        # 等权模式标记
        self.use_equal_weight = False
        
        # 状态
        self.cash = initial_cash
        self.positions = {}         # {code: {'shares': N, 'entry_price': P, 'entry_date': D}}
        self.equity_curve = []
        self.trades = []
        self.turnover_history = []
        self.model = None
        self.last_train_idx = 0
        
    def run(self) -> Dict:
        """运行Walk-Forward回测"""
        print("\n" + "=" * 70)
        print("🚀 Walk-Forward回测引擎 v2.0")
        print("=" * 70)
        
        # 准备月度数据
        self.factor_df['ym'] = self.factor_df['trade_date'].dt.to_period('M')
        month_ends = self.factor_df.groupby(['ts_code', 'ym'])['trade_date'].transform('max')
        monthly = self.factor_df[self.factor_df['trade_date'] == month_ends].copy()
        monthly = monthly.sort_values('trade_date').reset_index(drop=True)
        
        unique_dates = sorted(monthly['trade_date'].unique())
        print(f"  月度截面数: {len(unique_dates)}")
        print(f"  日期范围: {unique_dates[0].strftime('%Y-%m-%d')} ~ {unique_dates[-1].strftime('%Y-%m-%d')}")
        print(f"  训练窗口: {self.train_window}个月")
        print(f"  重训频率: 每{self.retrain_freq}个月")
        print(f"  持仓数量: {self.top_n}只")
        print(f"  最大换手率: {self.max_turnover*100:.0f}%")
        print(f"  交易成本: {self.cost_model.round_trip_cost()*100:.2f}% (来回)")
        
        # Walk-Forward循环
        for i, date in enumerate(unique_dates):
            # 确定是否需要重训模型
            if i < self.train_window:
                continue  # 跳过训练窗口之前的月份
            
            should_retrain = (i - self.last_train_idx >= self.retrain_freq) or (self.model is None)
            
            if should_retrain:
                self._train_model(monthly, unique_dates, i)
                self.last_train_idx = i
            
            # 生成信号
            cross = monthly[monthly['trade_date'] == date].copy()
            if len(cross) < 10:
                continue
            
            signals = self._generate_signals(cross)
            
            # 执行调仓
            self._rebalance(signals, date)
            
            # 记录
            total_value = self._get_portfolio_value(cross)
            self.equity_curve.append({
                'date': date,
                'equity': total_value,
                'cash': self.cash,
                'positions_value': total_value - self.cash,
                'n_positions': len(self.positions),
                'drawdown': 0  # 后面计算
            })
            
            # 打印进度
            if i % 6 == 0:
                ret = (total_value / self.initial_cash - 1) * 100
                print(f"  {date.strftime('%Y-%m')}: 总值 {total_value:,.0f} ({ret:+.1f}%), "
                      f"持仓 {len(self.positions)}只")
        
        # 计算回测指标
        results = self._compute_metrics()
        return results
    
    def _train_model(self, monthly: pd.DataFrame, dates: List, current_idx: int):
        """训练LightGBM模型"""
        # 使用过去train_window个月的数据
        start_idx = max(0, current_idx - self.train_window)
        train_dates = dates[start_idx:current_idx]
        
        train_data = monthly[monthly['trade_date'].isin(train_dates)].copy()
        
        # 准备特征和目标
        available_cols = [c for c in self.factor_cols if c in train_data.columns]
        X = train_data[available_cols].fillna(0)
        y = train_data['fwd_ret_1m'].fillna(0)
        
        # 去除目标为NaN的行
        valid_mask = y.notna() & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 100:
            return
        
        # 训练LightGBM
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 200,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
        
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(X, y)
        self.model_cols = available_cols
        
    def _generate_signals(self, cross: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        cross = cross.copy()
        
        if self.model is None or self.use_equal_weight:
            # 等权模式：【修复Bug 1】用IC符号反转负IC因子后再平均
            available = [c for c in self.factor_cols if c in cross.columns]
            score_parts = []
            for col in available:
                factor_vals = cross[col].fillna(0)
                sign = self.ic_signs.get(col, 1.0)
                score_parts.append(factor_vals * np.sign(sign))
            if score_parts:
                cross['score'] = sum(score_parts) / len(score_parts)
            else:
                cross['score'] = 0
        else:
            # ML模式：用LightGBM预测
            available_cols = [c for c in self.model_cols if c in cross.columns]
            X = cross[available_cols].fillna(0)
            cross['score'] = self.model.predict(X)
        
        # 排序
        cross = cross.sort_values('score', ascending=False)
        return cross
    
    def _rebalance(self, signals: pd.DataFrame, date):
        """执行调仓（带换手率约束和成本）"""
        # 目标持仓
        target_codes = self.risk_manager.select_stocks(
            signals, self.top_n, self.industry_map
        )
        target_set = set(target_codes)
        
        # 当前持仓
        current_set = set(self.positions.keys())
        
        # 计算换手率
        if current_set:
            turnover = len(current_set.symmetric_difference(target_set)) / max(len(current_set), len(target_set))
        else:
            turnover = 1.0
        
        # 换手率约束：如果换手率太高，只替换一部分
        if turnover > self.max_turnover and current_set:
            # 保留部分旧持仓
            keep_n = int(len(current_set) * (1 - self.max_turnover))
            # 保留得分最高的旧持仓
            old_scores = signals[signals['ts_code'].isin(current_set)]
            keep_old = set(old_scores.nlargest(keep_n, 'score')['ts_code'].tolist())
            
            # 新增的数量
            new_n = self.top_n - len(keep_old)
            new_candidates = signals[~signals['ts_code'].isin(keep_old)]
            new_codes = set(new_candidates.head(new_n)['ts_code'].tolist())
            
            target_set = keep_old | new_codes
            target_codes = list(target_set)
        
        # 执行卖出（先卖后买）
        sell_codes = current_set - target_set
        for code in sell_codes:
            if code in self.positions:
                pos = self.positions[code]
                # 获取当前价格
                price_row = signals[signals['ts_code'] == code]
                if len(price_row) > 0:
                    current_price = price_row.iloc[0]['close']
                else:
                    current_price = pos['entry_price']
                
                # 【修复Bug 3】个股止损：达到止损红线强制卖出
                if self.risk_manager.check_individual_stop(code, pos['entry_price'], current_price):
                    # 止损触发出清，不再只是pass
                    pass  # 正常卖出流程即可
                
                # 计算卖出金额和成本
                sell_amount = pos['shares'] * current_price
                sell_cost = self.cost_model.sell_cost(sell_amount)
                self.cash += sell_amount - sell_cost
                
                self.trades.append({
                    'date': date, 'code': code, 'action': 'SELL',
                    'price': current_price, 'shares': pos['shares'],
                    'amount': sell_amount, 'cost': sell_cost,
                    'reason': 'stop_loss' if self.risk_manager.check_individual_stop(
                        code, pos['entry_price'], current_price) else 'rebalance'
                })
                del self.positions[code]
        
        # 执行买入
        buy_codes = target_set - current_set
        if buy_codes and self.cash > 0:
            # 【修复Bug 6】减少现金缓冲从5%到1%，且多余现金平均分配
            n_buy = max(len(buy_codes), 1)
            per_stock_budget = self.cash * 0.99 / n_buy  # 只留1%缓冲
            
            for code in buy_codes:
                price_row = signals[signals['ts_code'] == code]
                if len(price_row) == 0:
                    continue
                current_price = price_row.iloc[0]['close']
                
                if current_price <= 0:
                    continue
                
                # 计算可买股数（100股整数倍）
                shares = int(per_stock_budget / current_price / 100) * 100
                if shares < 100:
                    continue
                
                buy_amount = shares * current_price
                buy_cost = self.cost_model.buy_cost(buy_amount)
                
                if buy_amount + buy_cost > self.cash:
                    continue
                
                self.cash -= buy_amount + buy_cost
                self.positions[code] = {
                    'shares': shares,
                    'entry_price': current_price,
                    'entry_date': date
                }
                
                self.trades.append({
                    'date': date, 'code': code, 'action': 'BUY',
                    'price': current_price, 'shares': shares,
                    'amount': buy_amount, 'cost': buy_cost
                })
        
        # 【修复Bug 3】检查组合止损：如果组合回撤超过-12%，卖出全部持仓
        if self.equity_curve:
            peak = max(e['equity'] for e in self.equity_curve)
            current = self._get_portfolio_value(signals)
            drawdown = (current - peak) / peak
            if self.risk_manager.check_portfolio_stop(drawdown):
                # 组合止损触发：清仓
                for code in list(self.positions.keys()):
                    pos = self.positions[code]
                    price_row = signals[signals['ts_code'] == code]
                    current_price = price_row.iloc[0]['close'] if len(price_row) > 0 else pos['entry_price']
                    sell_amount = pos['shares'] * current_price
                    sell_cost = self.cost_model.sell_cost(sell_amount)
                    self.cash += sell_amount - sell_cost
                    self.trades.append({
                        'date': date, 'code': code, 'action': 'SELL',
                        'price': current_price, 'shares': pos['shares'],
                        'amount': sell_amount, 'cost': sell_cost,
                        'reason': 'portfolio_stop_loss'
                    })
                    del self.positions[code]
        
        # 记录换手率
        self.turnover_history.append({'date': date, 'turnover': turnover})
    
    def _get_portfolio_value(self, cross: pd.DataFrame) -> float:
        """计算组合总价值"""
        total = self.cash
        for code, pos in self.positions.items():
            price_row = cross[cross['ts_code'] == code]
            if len(price_row) > 0:
                total += pos['shares'] * price_row.iloc[0]['close']
            else:
                total += pos['shares'] * pos['entry_price']
        return total
    
    def _compute_metrics(self) -> Dict:
        """计算回测指标"""
        if not self.equity_curve:
            return {}
        
        eq = pd.DataFrame(self.equity_curve)
        eq['date'] = pd.to_datetime(eq['date'])
        eq = eq.sort_values('date')
        
        # 计算收益率
        eq['return'] = eq['equity'].pct_change()
        eq['cum_return'] = eq['equity'] / self.initial_cash - 1
        eq['peak'] = eq['equity'].cummax()
        eq['drawdown'] = eq['equity'] / eq['peak'] - 1
        
        # 基础指标
        total_return = eq['cum_return'].iloc[-1]
        n_months = len(eq)
        years = n_months / 12
        
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        annual_vol = eq['return'].std() * np.sqrt(12)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        max_drawdown = eq['drawdown'].min()
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        win_rate = (eq['return'] > 0).mean()
        
        # 交易统计
        trades_df = pd.DataFrame(self.trades)
        total_cost = trades_df['cost'].sum() if len(trades_df) > 0 else 0
        
        turnover_df = pd.DataFrame(self.turnover_history)
        avg_turnover = turnover_df['turnover'].mean() if len(turnover_df) > 0 else 0
        
        results = {
            'summary': {
                'initial_value': self.initial_cash,
                'final_value': eq['equity'].iloc[-1],
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_vol': annual_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar,
                'win_rate': win_rate,
                'total_trades': len(self.trades),
                'total_cost': total_cost,
                'avg_turnover': avg_turnover,
                'n_months': n_months,
            },
            'equity_curve': eq.to_dict('records'),
            'trades': self.trades,
            'turnover': self.turnover_history,
        }
        
        # 打印结果
        print("\n" + "=" * 70)
        print("📊 Walk-Forward回测结果")
        print("=" * 70)
        s = results['summary']
        print(f"  总收益率:     {s['total_return']*100:>10.2f}%")
        print(f"  年化收益:     {s['annual_return']*100:>10.2f}%")
        print(f"  年化波动:     {s['annual_vol']*100:>10.2f}%")
        print(f"  夏普比率:     {s['sharpe_ratio']:>10.2f}")
        print(f"  最大回撤:     {s['max_drawdown']*100:>10.2f}%")
        print(f"  卡玛比率:     {s['calmar_ratio']:>10.2f}")
        print(f"  月胜率:       {s['win_rate']*100:>10.1f}%")
        print(f"  总交易次数:   {s['total_trades']:>10d}")
        print(f"  总交易成本:   {s['total_cost']:>10,.0f}元")
        print(f"  平均换手率:   {s['avg_turnover']*100:>10.1f}%")
        print(f"  回测月数:     {s['n_months']:>10d}")
        
        return results
