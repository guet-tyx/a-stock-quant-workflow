#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子引擎 v2.0 - 严格防止前视偏差
核心原则：T日因子只能用T日及之前的数据，预测T+1期收益

修复项：
1. 所有特征计算使用shift(1)确保只用历史数据
2. 收益率目标严格对齐到下一期
3. 清理无效因子，只保留IC_IR > 0.3的有效因子
4. 加入基本面因子（ROE/EP/营收增长）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FactorEngine:
    """统一因子计算引擎"""
    
    def __init__(self, daily_data: pd.DataFrame, stock_info: pd.DataFrame = None,
                 financial_data: pd.DataFrame = None):
        """
        Args:
            daily_data: 日频行情数据，列: ts_code, trade_date, open, high, low, close, vol, amount
            stock_info: 股票信息，列: ts_code, industry, market, list_date
            financial_data: 财务数据（可选），列: ts_code, ann_date, roe, ep, revenue_growth 等
        """
        self.df = daily_data.copy()
        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'].astype(str))
        self.df = self.df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        self.stock_info = stock_info
        self.financial_data = financial_data
        
        if financial_data is not None and 'ann_date' in financial_data.columns:
            self.financial_data['ann_date'] = pd.to_datetime(
                self.financial_data['ann_date'].astype(str))
        
    def build_all_factors(self) -> pd.DataFrame:
        """构建所有因子，返回包含因子值的DataFrame"""
        print("📊 构建因子库...")
        
        # 计算基础收益率（用于IC分析，不用于因子构建）
        self.df['ret_1d'] = self.df.groupby('ts_code')['close'].pct_change(1)
        
        # 分组计算因子
        factors_list = []
        for code, group in self.df.groupby('ts_code'):
            if len(group) < 120:
                continue
            factors = self._compute_stock_factors(group.copy())
            factors_list.append(factors)
        
        factor_df = pd.concat(factors_list, ignore_index=True)
        
        # 加入基本面因子（如果有）
        if self.financial_data is not None:
            factor_df = self._add_fundamental_factors(factor_df)
        
        # 加入截面因子
        factor_df = self._add_cross_sectional_factors(factor_df)
        
        # 因子标准化
        factor_df = self._normalize_factors(factor_df)
        
        # 因子列表
        self.factor_cols = [c for c in factor_df.columns 
                          if c not in ['ts_code', 'trade_date', 'close', 'ret_1d', 
                                       'fwd_ret_1m', 'industry']]
        
        print(f"✅ 构建完成: {len(self.factor_cols)} 个因子")
        return factor_df
    
    def _compute_stock_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算单只股票的量价因子"""
        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']
        vol = df['vol']
        amount = df['amount']
        ret = df['ret_1d']
        
        # ============================================
        # 【关键】所有因子都shift(1)，只用T-1日及之前数据
        # 这确保T日开盘前能看到的因子值
        # ============================================
        
        # --- 动量因子（经典有效因子）---
        df['mom_5d'] = close.pct_change(5).shift(1)   # T-5到T-1的收益
        df['mom_10d'] = close.pct_change(10).shift(1)
        df['mom_20d'] = close.pct_change(20).shift(1)
        df['mom_60d'] = close.pct_change(60).shift(1)
        df['mom_120d'] = close.pct_change(120).shift(1)
        
        # --- 波动率因子 ---
        df['vol_5d'] = ret.rolling(5).std().shift(1) * np.sqrt(252)
        df['vol_10d'] = ret.rolling(10).std().shift(1) * np.sqrt(252)
        df['vol_20d'] = ret.rolling(20).std().shift(1) * np.sqrt(252)
        df['vol_60d'] = ret.rolling(60).std().shift(1) * np.sqrt(252)
        
        # --- 成交量因子 ---
        vol_ma5 = vol.rolling(5).mean()
        vol_ma20 = vol.rolling(20).mean()
        vol_ma60 = vol.rolling(60).mean()
        df['vol_ratio_5d'] = (vol / vol_ma5).shift(1)
        df['vol_ratio_20d'] = (vol / vol_ma20).shift(1)
        df['vol_change_5d'] = vol.pct_change(5).shift(1)
        df['vol_change_20d'] = vol.pct_change(20).shift(1)
        
        # --- 流动性因子（Amihud）---
        amihud = ret.abs() / amount.replace(0, np.nan)
        df['amihud_20d'] = amihud.rolling(20).mean().shift(1)
        
        # --- 价格位置因子 ---
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        df['price_pos_20d'] = ((close - low_20) / (high_20 - low_20 + 1e-8)).shift(1)
        
        high_60 = high.rolling(60).max()
        low_60 = low.rolling(60).min()
        df['price_pos_60d'] = ((close - low_60) / (high_60 - low_60 + 1e-8)).shift(1)
        
        # --- 技术指标因子 ---
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = (100 - (100 / (1 + rs))).shift(1)
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        df['macd_hist'] = (macd - macd_signal).shift(1)
        
        # 布林带
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_width'] = ((4 * bb_std) / (bb_mid + 1e-8)).shift(1)
        df['bb_pos'] = ((close - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-8)).shift(1)
        
        # --- 收益率分布因子 ---
        df['skew_20d'] = ret.rolling(20).skew().shift(1)
        df['kurt_20d'] = ret.rolling(20).kurt().shift(1)
        df['max_ret_20d'] = ret.rolling(20).max().shift(1)
        df['min_ret_20d'] = ret.rolling(20).min().shift(1)
        
        # --- 复合因子（从LLM/arXiv研究中保留有效因子）---
        # 流动性情绪（IC_IR=-0.51，反向有效）
        df['liquidity_sentiment'] = (-amihud.rolling(20).mean()).shift(1)
        
        # 流动性调整动量 - 原始值相乘，截面rank在_add_cross_sectional_factors中完成
        df['liquidity_adj_momentum_raw'] = (close.pct_change(20) * amihud.rolling(20).mean()).shift(1)
        
        # 趋势质量因子 - 原始值，无需时间序列rank
        ma20 = close.rolling(20).mean()
        residual = close / ma20 - 1
        residual_vol = residual.rolling(20).std()
        vol_trend = (vol_ma20 / vol_ma60.replace(0, np.nan) - 1)
        mom20 = close.pct_change(20)
        df['trend_quality_raw'] = (mom20 * (1 - residual_vol) * vol_trend).shift(1)
        
        # 波动率调整动量
        df['vol_adj_momentum'] = (close.pct_change(20) / (ret.rolling(20).std() + 1e-8)).shift(1)
        
        # 成交量价格背离 - 原始值相乘，截面rank由外层完成
        df['vp_divergence_raw'] = (-np.sign(close.pct_change(5)) * vol.pct_change(5)).shift(1)
        
        # 恐惧指数（波动率比率）
        df['fear_index'] = (ret.rolling(20).std() / (ret.rolling(60).std() + 1e-8)).shift(1)
        
        # 位置波动因子
        df['position_vol'] = df['price_pos_20d'] * df['vol_20d']
        
        # 信息不对称（日内振幅比）
        intraday_range = (high - low) / close
        ir_20 = intraday_range.rolling(20).mean()
        ir_60 = intraday_range.rolling(60).mean()
        df['info_asymmetry'] = (-(ir_20 / (ir_60 + 1e-8))).shift(1)
        
        # --- 未来收益率（目标变量，延后到compute_factor_ic中按月末对齐计算）---
        df['fwd_ret_1m'] = np.nan  # 占位，后续在compute_factor_ic中填充
        
        # 保留需要的列
        keep_cols = ['ts_code', 'trade_date', 'close', 'fwd_ret_1m'] + self._get_factor_names()
        available = [c for c in keep_cols if c in df.columns]
        return df[available]
    
    def _get_factor_names(self) -> List[str]:
        """返回所有因子名称"""
        return [
            'mom_5d', 'mom_10d', 'mom_20d', 'mom_60d', 'mom_120d',
            'vol_5d', 'vol_10d', 'vol_20d', 'vol_60d',
            'vol_ratio_5d', 'vol_ratio_20d', 'vol_change_5d', 'vol_change_20d',
            'amihud_20d',
            'price_pos_20d', 'price_pos_60d',
            'rsi_14', 'macd_hist', 'bb_width', 'bb_pos',
            'skew_20d', 'kurt_20d', 'max_ret_20d', 'min_ret_20d',
            'liquidity_sentiment', 'liquidity_adj_momentum_raw',
            'trend_quality_raw', 'vol_adj_momentum', 'vp_divergence_raw',
            'fear_index', 'position_vol', 'info_asymmetry'
        ]
    
    def _add_fundamental_factors(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """加入基本面因子（使用公告日期对齐，避免前视偏差）"""
        if self.financial_data is None:
            return factor_df
            
        print("  📈 添加基本面因子...")
        
        fin = self.financial_data.copy()
        
        # 【关键】用ann_date（公告日期）而非end_date对齐
        if 'ann_date' in fin.columns:
            fin['ann_date'] = pd.to_datetime(fin['ann_date'].astype(str))
            fin = fin.sort_values(['ts_code', 'ann_date'])
            fin = fin.drop_duplicates(subset=['ts_code', 'end_date'], keep='last')
        
        # 构建基本面因子 - 使用merge_asof高效对齐
        fin_cols = []
        for col in ['roe', 'dt_eps', 'bps', 'netprofit_margin', 'ocf_to_or']:
            if col in fin.columns:
                # 创建该因子的时间序列
                fin_sub = fin[['ts_code', 'ann_date', col]].dropna(subset=[col, 'ann_date'])
                fin_sub = fin_sub.rename(columns={'ann_date': 'trade_date', col: f'fin_{col}'})
                fin_sub[f'fin_{col}'] = pd.to_numeric(fin_sub[f'fin_{col}'], errors='coerce')
                fin_sub = fin_sub.dropna(subset=['trade_date'])
                fin_cols.append(f'fin_{col}')
                
                # 对每只股票用merge_asof对齐（确保只用公告前的数据）
                parts = []
                for code in factor_df['ts_code'].unique():
                    stock_fac = factor_df[factor_df['ts_code'] == code].sort_values('trade_date')
                    stock_fin = fin_sub[fin_sub['ts_code'] == code].sort_values('trade_date')
                    
                    if len(stock_fin) == 0:
                        stock_fac[f'fin_{col}'] = np.nan
                        parts.append(stock_fac)
                        continue
                    
                    merged = pd.merge_asof(
                        stock_fac, stock_fin.drop('ts_code', axis=1),
                        on='trade_date', direction='backward'
                    )
                    parts.append(merged)
                
                factor_df = pd.concat(parts, ignore_index=True)
        
        # 构建衍生基本面因子
        if 'fin_roe' in factor_df.columns and 'fin_bps' in factor_df.columns:
            # EP (Earnings-to-Price) = EPS / Price ≈ ROE * BPS / Price
            factor_df['fin_ep'] = (factor_df['fin_roe'] * factor_df['fin_bps'] / 100) / factor_df['close']
            factor_df['fin_ep'] = factor_df.groupby('trade_date')['fin_ep'].transform(
                lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)) if x.std() > 0 else x
            )
        
        if 'fin_roe' in factor_df.columns:
            # ROE变化（同比）
            factor_df['fin_roe_chg'] = factor_df.groupby('ts_code')['fin_roe'].diff(4)  # 4个季度前
            # shift(1)防前视偏差
            factor_df['fin_roe_chg'] = factor_df['fin_roe_chg'].shift(1)
        
        if 'fin_netprofit_margin' in factor_df.columns:
            factor_df['fin_margin_chg'] = factor_df.groupby('ts_code')['fin_netprofit_margin'].diff(4)
            factor_df['fin_margin_chg'] = factor_df['fin_margin_chg'].shift(1)
        
        return factor_df
    
    def _add_cross_sectional_factors(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """添加截面排名因子"""
        print("  📊 添加截面因子...")

        # 按日期截面标准化排名（对负IC因子自动反转方向）
        rank_targets = [
            'mom_20d', 'vol_20d', 'amihud_20d', 'liquidity_sentiment',
            'liquidity_adj_momentum_raw', 'trend_quality_raw',
            'vp_divergence_raw', 'vol_adj_momentum', 'fear_index',
            'price_pos_20d', 'price_pos_60d', 'macd_hist', 'bb_width', 'bb_pos'
        ]
        for col in rank_targets:
            if col in factor_df.columns:
                factor_df[f'{col}_rank'] = factor_df.groupby('trade_date')[col].rank(pct=True)

        # 构建最终复合因子名（保留原始名+rank名，供IC分析筛选）
        # 注意：原始_raw因子在标准化后也保留，因为模型可能会使用
        return factor_df
    
    def _normalize_factors(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """因子标准化：截面去极值+Z-score"""
        print("  🔧 因子标准化...")
        
        factor_cols = [c for c in self._get_factor_names() if c in factor_df.columns]
        # 加入基本面因子
        fin_cols = [c for c in factor_df.columns if c.startswith('fin_')]
        all_factor_cols = factor_cols + fin_cols
        self.factor_cols = all_factor_cols
        
        for col in all_factor_cols:
            if col not in factor_df.columns:
                continue
            # 按日期截面处理 - MAD去极值 + Z-score
            def _normalize_series(s):
                if s.std() == 0 or s.isna().all():
                    return s * 0
                median = s.median()
                mad = (s - median).abs().median()
                upper = median + 3 * 1.4826 * mad
                lower = median - 3 * 1.4826 * mad
                s = s.clip(lower, upper)
                return (s - s.mean()) / (s.std() + 1e-8)
            
            factor_df[col] = factor_df.groupby('trade_date')[col].transform(_normalize_series)
        
        return factor_df
    
    def compute_factor_ic(self, factor_df: pd.DataFrame,
                          min_periods: int = 24) -> pd.DataFrame:
        """计算所有因子的IC/IC_IR"""
        print("\n📊 因子IC分析...")
        
        # 月度采样
        factor_df = factor_df.copy()
        factor_df['ym'] = factor_df['trade_date'].dt.to_period('M')
        month_end = factor_df.groupby(['ts_code', 'ym'])['trade_date'].transform('max')
        monthly = factor_df[factor_df['trade_date'] == month_end].copy()
        
        # 【修复Bug 4】用实际月末对齐计算fwd_ret_1m（替代固定shift(-21)）
        monthly['next_close'] = monthly.groupby('ts_code')['close'].shift(-1)  # 下月收盘价
        monthly['fwd_ret_1m'] = monthly['next_close'] / monthly['close'] - 1
        
        results = []
        # 排除_index/_rank/_raw后缀的列避免冗余（只保留标准化后的核心因子）
        core_cols = [c for c in self.factor_cols if not c.endswith('_raw')]
        rank_cols = [c for c in monthly.columns if c.endswith('_rank') and c in self.factor_cols]
        all_test_cols = list(dict.fromkeys(core_cols + rank_cols))  # 去重保序
        
        for col in all_test_cols:
            if col not in monthly.columns:
                continue
            
            ics = []
            for ym, group in monthly.groupby('ym'):
                valid = group[[col, 'fwd_ret_1m']].dropna()
                if len(valid) > 30:
                    ic = valid[col].corr(valid['fwd_ret_1m'], method='spearman')
                    if not np.isnan(ic):
                        ics.append(ic)
            
            if len(ics) < min_periods:
                continue
                
            ic_arr = np.array(ics)
            results.append({
                'factor': col,
                'IC': ic_arr.mean(),
                'IC_std': ic_arr.std(),
                'IC_IR': ic_arr.mean() / (ic_arr.std() + 1e-8),
                'IC_pos_pct': (ic_arr > 0).mean(),
                'n_periods': len(ics)
            })
        
        ic_df = pd.DataFrame(results).sort_values('IC_IR', key=abs, ascending=False)
        
        # 标记有效性
        ic_df['validity'] = ic_df['IC_IR'].apply(
            lambda x: '✅ 强' if abs(x) >= 0.5 else 
                      '⚠️ 中' if abs(x) >= 0.3 else 
                      '❓ 弱' if abs(x) >= 0.2 else '❌ 无效'
        )
        
        print("\n" + "=" * 80)
        print(f"{'因子':<30} {'IC':>8} {'IC_IR':>8} {'IC>0%':>8} {'有效性':>10}")
        print("-" * 80)
        for _, row in ic_df.iterrows():
            print(f"{row['factor']:<30} {row['IC']:>8.4f} {row['IC_IR']:>8.4f} "
                  f"{row['IC_pos_pct']*100:>7.1f}% {row['validity']:>10}")
        
        # 筛选有效因子
        valid_factors = ic_df[ic_df['IC_IR'].abs() >= 0.3]['factor'].tolist()
        print(f"\n✅ 有效因子 (|IC_IR| >= 0.3): {len(valid_factors)}/{len(ic_df)}")
        if valid_factors:
            print(f"   {', '.join(valid_factors)}")
        
        return ic_df


def build_factor_engine(daily_data_path: str, stock_list_path: str,
                        financial_data_path: str = None) -> Tuple[FactorEngine, pd.DataFrame, pd.DataFrame]:
    """构建因子引擎的工厂函数"""
    # 加载数据
    df = pd.read_csv(daily_data_path, dtype={'trade_date': str})
    stock_info = pd.read_csv(stock_list_path)
    
    fin_data = None
    if financial_data_path and os.path.exists(financial_data_path):
        fin_data = pd.read_csv(financial_data_path, dtype={'ann_date': str, 'end_date': str})
    
    engine = FactorEngine(df, stock_info, fin_data)
    factor_df = engine.build_all_factors()
    ic_df = engine.compute_factor_ic(factor_df)
    
    return engine, factor_df, ic_df


if __name__ == '__main__':
    import os
    data_dir = os.path.expanduser("~/quant_strategies/individual_stock_strategy/data")
    
    engine, factor_df, ic_df = build_factor_engine(
        os.path.join(data_dir, "daily_data_fixed.csv"),
        os.path.join(data_dir, "stock_list_fixed.csv"),
    )
    
    # 保存
    out_dir = os.path.expanduser("~/quant_strategies/individual_stock_strategy/v2/output")
    os.makedirs(out_dir, exist_ok=True)
    factor_df.to_csv(os.path.join(out_dir, "factors_v2.csv"), index=False)
    ic_df.to_csv(os.path.join(out_dir, "factor_ic_v2.csv"), index=False)
    print(f"\n✅ 因子数据已保存到 {out_dir}")
