import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("LLM因子挖掘与测试")
print("=" * 60)

data_path = os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/daily_data_fixed.csv")
df = pd.read_csv(data_path, dtype={"trade_date": str})
df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

print(f"数据范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
print(f"股票数量: {df['ts_code'].nunique()}")
print(f"总记录数: {len(df)}")

# 基础特征
df["ret_1d"] = df.groupby("ts_code")["close"].pct_change(1)
df["ret_5d"] = df.groupby("ts_code")["close"].pct_change(5)
df["vol_20d"] = df.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(20).std())
df["vol_60d"] = df.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60).std())
df["mom_5d"] = df.groupby("ts_code")["close"].pct_change(5)
df["mom_20d"] = df.groupby("ts_code")["close"].pct_change(20)
df["mom_60d"] = df.groupby("ts_code")["close"].pct_change(60)
df["mom_120d"] = df.groupby("ts_code")["close"].pct_change(120)
df["vol_ma20"] = df.groupby("ts_code")["vol"].transform(lambda x: x.rolling(20).mean())
df["vol_ma60"] = df.groupby("ts_code")["vol"].transform(lambda x: x.rolling(60).mean())
df["vol_change_5d"] = df.groupby("ts_code")["vol"].pct_change(5)
df["intraday_range"] = (df["high"] - df["low"]) / df["close"]
df["amihud"] = df["ret_1d"].abs() / df["amount"].replace(0, np.nan)
df["ma5"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(5).mean())
df["ma20"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).mean())
df["ma60"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(60).mean())
df["min_20d"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).min())
df["max_20d"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).max())
df["up"] = (df["ret_1d"] > 0).astype(int)
df["consecutive_up"] = df.groupby("ts_code")["up"].transform(
    lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
)
df["gap"] = (df["open"] / df.groupby("ts_code")["close"].shift(1) - 1).abs()
df["vol_spike"] = df["vol"] / df["vol_ma20"].replace(0, np.nan)

factors = pd.DataFrame()
factors["ts_code"] = df["ts_code"]
factors["trade_date"] = df["trade_date"]

# 13个LLM风格因子
factors["fear_index"] = df["vol_20d"] / df["vol_60d"].replace(0, np.nan)
factors["greed_index"] = df["consecutive_up"] * df["ret_5d"].clip(-0.2, 0.2)
factors["attention_factor"] = df["vol_ma20"] / df["vol_ma60"].replace(0, np.nan)
factors["momentum_sentiment"] = (
    0.4 * df["mom_5d"].rank(pct=True) + 0.3 * df["mom_20d"].rank(pct=True) +
    0.2 * df["mom_60d"].rank(pct=True) + 0.1 * df["mom_120d"].rank(pct=True)
)
factors["reversal_sentiment"] = (df["close"] - df["min_20d"]) / (df["max_20d"] - df["min_20d"]).replace(0, np.nan)
factors["volatility_sentiment"] = -df["vol_20d"].rank(pct=True)
factors["vp_divergence"] = -np.sign(df["mom_5d"]) * df["vol_change_5d"].rank(pct=True)
factors["info_asymmetry"] = -(
    df.groupby("ts_code")["intraday_range"].transform(lambda x: x.rolling(20).mean()) /
    df.groupby("ts_code")["intraday_range"].transform(lambda x: x.rolling(60).mean()).replace(0, np.nan)
).rank(pct=True)
factors["liquidity_sentiment"] = -df.groupby("ts_code")["amihud"].transform(
    lambda x: x.rolling(20).mean()
).rank(pct=True)
factors["composite_sentiment"] = (
    0.15 * factors["fear_index"].rank(pct=True) + 0.10 * factors["greed_index"].rank(pct=True) +
    0.15 * factors["attention_factor"].rank(pct=True) + 0.15 * factors["momentum_sentiment"] +
    0.10 * factors["reversal_sentiment"].rank(pct=True) + 0.10 * factors["volatility_sentiment"] +
    0.10 * factors["vp_divergence"].rank(pct=True) + 0.05 * factors["info_asymmetry"] +
    0.10 * factors["liquidity_sentiment"]
)
bullish_alignment = ((df["ma5"] > df["ma20"]) & (df["ma20"] > df["ma60"])).astype(int)
vol_ratio = df["vol_ma20"] / df["vol_ma60"].replace(0, np.nan)
factors["llm_sentiment_score"] = 0.6 * bullish_alignment + 0.4 * vol_ratio.rank(pct=True)
factors["event_driven"] = df["gap"].rank(pct=True) * df["vol_spike"].rank(pct=True)
fear_sign = np.sign(factors["fear_index"] - 1)
vol_sign = np.sign(factors["volatility_sentiment"])
mom_sign = np.sign(factors["momentum_sentiment"] - 0.5)
factors["sentiment_consensus"] = (fear_sign + vol_sign + mom_sign).abs() / 3

factor_cols = [
    "fear_index", "greed_index", "attention_factor", "momentum_sentiment",
    "reversal_sentiment", "volatility_sentiment", "vp_divergence",
    "info_asymmetry", "liquidity_sentiment", "composite_sentiment",
    "llm_sentiment_score", "event_driven", "sentiment_consensus"
]

print(f"\n构建了 {len(factor_cols)} 个LLM风格因子")

# IC分析 - 月度采样
df["fwd_ret_20d"] = df.groupby("ts_code")["close"].pct_change(20).shift(-20)
df["ym"] = df["trade_date"].dt.to_period("M")
month_end = df.groupby(["ts_code", "ym"])["trade_date"].transform("max")
monthly_mask = df["trade_date"] == month_end
monthly = df[monthly_mask].copy()
monthly = monthly.merge(factors[factor_cols + ["ts_code", "trade_date"]], on=["ts_code", "trade_date"], how="left")

print(f"月度截面数: {monthly['ym'].nunique()}")

ic_results = {}
for col in factor_cols:
    monthly_ic = []
    for ym, group in monthly.groupby("ym"):
        valid = group[[col, "fwd_ret_20d"]].dropna()
        if len(valid) > 20:
            ic = valid[col].corr(valid["fwd_ret_20d"])
            if not np.isnan(ic):
                monthly_ic.append(ic)
    if len(monthly_ic) < 5:
        continue
    ic_arr = np.array(monthly_ic)
    ic_mean = ic_arr.mean()
    ic_std = ic_arr.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_pos_pct = (ic_arr > 0).mean()
    ic_results[col] = {"IC": round(ic_mean, 4), "IC_IR": round(ic_ir, 4), "IC>0%": round(ic_pos_pct * 100, 2)}

print("\n" + "=" * 75)
print("因子IC分析结果:")
print("=" * 75)
print(f"{'因子名称':<25} {'IC':>8} {'IC_IR':>8} {'IC>0%':>8} {'有效性':>10}")
print("-" * 75)

valid_factors = []
for col, res in sorted(ic_results.items(), key=lambda x: abs(x[1]["IC_IR"]), reverse=True):
    if abs(res["IC_IR"]) >= 0.3:
        validity = "✅ 有效"
        valid_factors.append(col)
    elif abs(res["IC_IR"]) >= 0.2:
        validity = "⚠️ 边缘"
    else:
        validity = "❌ 无效"
    print(f"{col:<25} {res['IC']:>8.4f} {res['IC_IR']:>8.4f} {res['IC>0%']:>7.1f}% {validity:>10}")

print(f"\n有效因子: {len(valid_factors)}/{len(factor_cols)}")
if valid_factors:
    print(f"有效因子列表: {', '.join(valid_factors)}")

factors.to_csv(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/llm_factors.csv"), index=False)
pd.DataFrame(ic_results).T.to_csv(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/llm_factor_ic.csv"))
with open(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/llm_ic_results.json"), "w") as f:
    json.dump(ic_results, f, indent=2)

print("\n✅ LLM因子IC分析完成")
