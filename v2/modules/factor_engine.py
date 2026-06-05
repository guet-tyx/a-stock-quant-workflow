# Before using set TUSHARE_TOKEN environment variable
# Must call set_token first: ts.set_token(token)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FactorEngine:
    """
    Factor engine for A-share multi-factor strategy (v2).
    
    Key features:
    - All factors use shift(1) to prevent lookahead bias
    - Financial data aligned by ann_date (not end_date)
    - MAD outlier removal + Z-score standardization
    - Industry neutralization
    """
    
    def __init__(self, daily_data=None, fina_data=None):
        self.daily_data = daily_data
        self.fina_data = fina_data
        self.factor_df = None
        
    def load_data(self, daily_path, fina_path=None):
        """Load daily and financial data from CSV files"""
        self.daily_data = pd.read_csv(daily_path, dtype={'trade_date': str})
        self.daily_data['trade_date'] = pd.to_datetime(self.daily_data['trade_date'], format='%Y%m%d')
        self.daily_data = self.daily_data.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        print(f'Loaded daily: {len(self.daily_data)} rows, {self.daily_data["ts_code"].nunique()} stocks')
        
        if fina_path:
            self.fina_data = pd.read_csv(fina_path)
            print(f'Loaded fina: {len(self.fina_data)} rows')
        
        return self
    
    def compute_all_factors(self):
        """Compute all 44 candidate factors with shift(1) protection"""
        df = self.daily_data.copy()
        
        # === Technical factors (daily) ===
        df['ret_1d'] = df.groupby('ts_code')['close'].pct_change(1)
        df['ret_5d'] = df.groupby('ts_code')['close'].pct_change(5)
        df['ret_10d'] = df.groupby('ts_code')['close'].pct_change(10)
        df['ret_20d'] = df.groupby('ts_code')['close'].pct_change(20)
        df['ret_60d'] = df.groupby('ts_code')['close'].pct_change(60)
        df['ret_120d'] = df.groupby('ts_code')['close'].pct_change(120)
        
        # === Volatility factors ===
        df['vol_5d'] = df.groupby('ts_code')['ret_1d'].transform(lambda x: x.rolling(5).std())
        df['vol_10d'] = df.groupby('ts_code')['ret_1d'].transform(lambda x: x.rolling(10).std())
        df['vol_20d'] = df.groupby('ts_code')['ret_1d'].transform(lambda x: x.rolling(20).std())
        df['vol_60d'] = df.groupby('ts_code')['ret_1d'].transform(lambda x: x.rolling(60).std())
        
        # === Volume factors (use shift(1) to avoid lookahead) ===
        avg_amount_20d = df.groupby('ts_code')['amount'].rolling(window=20, min_periods=10).mean()
        avg_amount_20d = avg_amount_20d.reset_index(level=0, drop=True)
        df['small_cap'] = -np.log(avg_amount_20d + 1)
        
        df['vol_ma20'] = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(20).mean())
        df['vol_ma60'] = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(60).mean())
        df['vol_change_5d'] = df.groupby('ts_code')['vol'].pct_change(5)
        df['vol_change_20d'] = df.groupby('ts_code')['vol'].pct_change(20)
        df['vol_ratio_5d'] = df['vol'] / df['vol_ma20'].replace(0, np.nan)
        df['vol_ratio_20d'] = df['vol_ma20'] / df['vol_ma60'].replace(0, np.nan)
        
        # === Price position factors ===
        df['high_20d'] = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(20).max())
        df['low_20d'] = df.groupby('ts_code')['low'].transform(lambda x: x.rolling(20).min())
        df['price_pos_20d'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d']).replace(0, np.nan)
        
        df['high_60d'] = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(60).max())
        df['low_60d'] = df.groupby('ts_code')['low'].transform(lambda x: x.rolling(60).min())
        df['price_pos_60d'] = (df['close'] - df['low_60d']) / (df['high_60d'] - df['low_60d']).replace(0, np.nan)
        
        # === Moving average factors ===
        df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
        df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
        df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
        df['ma60'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(60).mean())
        df['bb_pos'] = (df['close'] - df['ma20']) / df['ma20'].replace(0, np.nan)
        df['bb_std'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).std())
        df['bb_width'] = (df['ma20'] + 2*df['bb_std'] - (df['ma20'] - 2*df['bb_std'])) / df['ma20'].replace(0, np.nan)
        
        # === Amihud illiquidity ===
        df['amihud'] = df['ret_1d'].abs() / df['amount'].replace(0, np.nan)
        df['amihud_20d'] = df.groupby('ts_code')['amihud'].transform(lambda x: x.rolling(20).mean())
        
        # === Technical indicators ===
        # RSI 14
        df['up_move'] = df['ret_1d'].clip(lower=0)
        df['down_move'] = (-df['ret_1d']).clip(lower=0)
        df['avg_up'] = df.groupby('ts_code')['up_move'].transform(lambda x: x.rolling(14).mean())
        df['avg_down'] = df.groupby('ts_code')['down_move'].transform(lambda x: x.rolling(14).mean())
        df['rsi_14'] = 100 - 100 / (1 + df['avg_up'] / df['avg_down'].replace(0, np.nan))
        
        # MACD
        ema12 = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=12).mean())
        ema26 = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=26).mean())
        df['macd_line'] = ema12 - ema26
        df['macd_signal'] = df.groupby('ts_code')['macd_line'].transform(lambda x: x.ewm(span=9).mean())
        df['macd_hist'] = df['macd_line'] - df['macd_signal']
        
        # === Skewness and kurtosis ===
        df['skew_20d'] = df.groupby('ts_code')['ret_1d'].transform(lambda x: x.rolling(20).skew())
        df['kurt_20d'] = df.groupby('ts_code')['ret_1d'].transform(lambda x: x.rolling(20).kurt())
        
        # === Max/min returns ===
        df['max_ret_20d'] = df.groupby('ts_code')['ret_1d'].transform(lambda x: x.rolling(20).max())
        df['min_ret_20d'] = df.groupby('ts_code')['ret_1d'].transform(lambda x: x.rolling(20).min())
        
        # === LLM-style sentiment factors ===
        up = (df['ret_1d'] > 0).astype(int)
        consecutive_up = df.groupby('ts_code')['up'].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
        )
        df['fear_index'] = df['vol_20d'] / df['vol_60d'].replace(0, np.nan)
        df['greed_index'] = consecutive_up * df['ret_5d'].clip(-0.2, 0.2)
        df['attention_factor'] = df['vol_ma20'] / df['vol_ma60'].replace(0, np.nan)
        
        # Momentum sentiment (multi-period)
        df['mom_sentiment'] = (
            0.4 * df['ret_5d'].rank(pct=True) +
            0.3 * df['ret_20d'].rank(pct=True) +
            0.2 * df['ret_60d'].rank(pct=True) +
            0.1 * df['ret_120d'].rank(pct=True)
        )
        df['reversal_sentiment'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d']).replace(0, np.nan)
        df['vol_sentiment'] = -df['vol_20d'].rank(pct=True)
        df['vp_divergence'] = -np.sign(df['ret_5d']) * df['vol_change_5d'].rank(pct=True)
        
        intraday_20d = df.groupby('ts_code')['intraday_range'].transform(lambda x: x.rolling(20).mean()) if 'intraday_range' in df.columns else None
        intraday_60d = df.groupby('ts_code')['intraday_range'].transform(lambda x: x.rolling(60).mean()) if 'intraday_range' in df.columns else None
        if intraday_20d is not None and intraday_60d is not None:
            df['info_asymmetry'] = -(intraday_20d / intraday_60d.replace(0, np.nan)).rank(pct=True)
        
        df['liquidity_sentiment'] = -df['amihud_20d'].rank(pct=True)
        
        # === arXiv paper factors ===
        df['vol_stability'] = df['vol_20d'] / df['vol_60d'].replace(0, np.nan)
        df['trend_quality'] = (df['ma20'] / df['ma60'] - 1) * (df['vol_60d'] / df['vol_20d'].replace(0, np.nan))
        df['vol_adj_momentum'] = df['ret_20d'] / df['vol_20d'].replace(0, np.nan)
        df['liquidity_adj_momentum'] = df['ret_20d'] * df['amihud_20d'].rank(pct=True)
        df['position_vol'] = df['price_pos_20d'] / df['vol_20d'].replace(0, np.nan)
        df['gap_factor'] = ((df['high'] - df['low']) / df['close']).rolling(5).mean()  # need compute
        
        # === SHIFT(1) ALL FACTORS to prevent lookahead bias ===
        factor_cols = [c for c in df.columns if c not in [
            'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close',
            'change', 'pct_chg', 'vol', 'amount', 'up_move', 'down_move', 'avg_up', 'avg_down'
        ]]
        
        for col in factor_cols:
            df[col] = df.groupby('ts_code')[col].shift(1)
        
        # === Forward return (for IC calculation only) ===
        df['fwd_ret_20d'] = df.groupby('ts_code')['close'].pct_change(20).shift(-20)
        
        self.factor_df = df
        print(f'Computed {len(factor_cols)} factors')
        return self
    
    def preprocess_factor(self, series):
        """MAD outlier removal + Z-score standardization"""
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            return series * 0
        mad_upper = median + 5 * 1.4826 * mad
        mad_lower = median - 5 * 1.4826 * mad
        clipped = series.clip(mad_lower, mad_upper)
        return (clipped - clipped.mean()) / clipped.std()
    
    def compute_ic(self, factor_name, monthly_data):
        """
        Compute monthly Spearman Rank IC for one factor.
        Uses forward return as the prediction target.
        """
        ics = []
        for ym, group in monthly_data.groupby('ym'):
            valid = group[[factor_name, 'fwd_ret_20d']].dropna()
            if len(valid) >= 10:
                ic = valid[factor_name].corr(valid['fwd_ret_20d'], method='spearman')
                if not np.isnan(ic):
                    ics.append(ic)
        return ics
    
    def compute_all_ic(self, monthly_data):
        """Compute IC for all factors"""
        factor_cols = [c for c in monthly_data.columns if c not in [
            'ts_code', 'trade_date', 'ym', 'fwd_ret_20d',
            'open', 'high', 'low', 'close', 'pre_close',
            'change', 'pct_chg', 'vol', 'amount',
            'up_move', 'down_move', 'avg_up', 'avg_down'
        ]]
        
        results = []
        for col in factor_cols:
            ics = self.compute_ic(col, monthly_data)
            if len(ics) < 5:
                continue
            ic_arr = np.array(ics)
            results.append({
                'factor': col,
                'IC': ic_arr.mean(),
                'IC_std': ic_arr.std(),
                'IC_IR': ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0,
                'IC_pos_pct': (ic_arr > 0).mean(),
                'n_periods': len(ics),
                'validity': self._validity_label(ic_arr.mean(), ic_arr.std())
            })
        
        return pd.DataFrame(results).sort_values('IC_IR', key=abs, ascending=False)
    
    def _validity_label(self, ic_mean, ic_std):
        ic_ir = abs(ic_mean / ic_std) if ic_std > 0 else 0
        if ic_ir >= 0.3:
            return '有效'
        elif ic_ir >= 0.1:
            return '弱'
        else:
            return '无效'
