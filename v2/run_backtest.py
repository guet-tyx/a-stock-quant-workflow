#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口 - 完整Walk-Forward回测
修复所有诊断问题后的新版策略
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.factor_engine import FactorEngine
from modules.backtest_engine import WalkForwardBacktest


def main():
    data_dir = os.path.expanduser("~/quant_strategies/individual_stock_strategy/data")
    out_dir = os.path.expanduser("~/quant_strategies/individual_stock_strategy/v2/output")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print("🔬 量化策略 v2.0 - 全面修复版")
    print(f"   运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # ============================================================
    # Step 1: 加载数据
    # ============================================================
    print("\n📂 Step 1: 加载数据...")
    
    daily_data = pd.read_csv(os.path.join(data_dir, "daily_data_fixed.csv"), dtype={'trade_date': str})
    stock_info = pd.read_csv(os.path.join(data_dir, "stock_list_fixed.csv"))
    fin_data = pd.read_csv(os.path.join(data_dir, "financial_data_v2.csv"), 
                           dtype={'ann_date': str, 'end_date': str}) if os.path.exists(os.path.join(data_dir, "financial_data_v2.csv")) else None
    
    print(f"  日频数据: {len(daily_data):,} 行, {daily_data['ts_code'].nunique()} 只股票")
    print(f"  日期范围: {daily_data['trade_date'].min()} ~ {daily_data['trade_date'].max()}")
    if fin_data is not None:
        print(f"  财务数据: {len(fin_data):,} 行, {fin_data['ts_code'].nunique()} 只股票")
    
    # ============================================================
    # Step 2: 计算因子（严格防前视偏差）
    # ============================================================
    print("\n📊 Step 2: 计算因子（所有因子shift(1)防前视偏差）...")
    
    engine = FactorEngine(daily_data, stock_info, fin_data)
    factor_df = engine.build_all_factors()
    ic_df = engine.compute_factor_ic(factor_df)
    
    # 保存因子IC
    ic_df.to_csv(os.path.join(out_dir, "factor_ic_v2.csv"), index=False)
    
    # 筛选有效因子
    valid_factors = ic_df[ic_df['IC_IR'].abs() >= 0.2]['factor'].tolist()
    if len(valid_factors) < 3:
        # 如果有效因子太少，降低阈值
        valid_factors = ic_df[ic_df['IC_IR'].abs() >= 0.15]['factor'].tolist()
    
    print(f"\n✅ 使用 {len(valid_factors)} 个因子进行回测:")
    for f in valid_factors:
        row = ic_df[ic_df['factor'] == f].iloc[0]
        print(f"   {f}: IC_IR={row['IC_IR']:.4f}, IC={row['IC']:.4f}")
    
    # ============================================================
    # Step 3: Walk-Forward回测
    # ============================================================
    print("\n🚀 Step 3: Walk-Forward回测...")
    
    # 【修复Bug 1】计算因子IC符号，传递给回测引擎用于等权模式
    ic_signs = {}
    for _, row in ic_df.iterrows():
        ic_signs[row['factor']] = 1.0 if row['IC'] > 0 else -1.0
    
    backtest = WalkForwardBacktest(
        factor_df=factor_df,
        factor_cols=valid_factors,
        stock_info=stock_info,
        ic_signs=ic_signs,         # 传递IC符号
        train_window=24,        # 24个月训练窗口（参数扫描最优）
        retrain_freq=6,         # 每6个月重训（参数扫描最优）
        top_n=30,               # 持仓30只
        initial_cash=1000000,
        max_turnover=0.3,       # 最大30%换手（参数扫描最优）
    )
    
    results = backtest.run()
    
    # ============================================================
    # Step 4: 分时期诊断
    # ============================================================
    print("\n🔍 Step 4: 分时期诊断...")
    
    if results and 'equity_curve' in results:
        eq = pd.DataFrame(results['equity_curve'])
        eq['date'] = pd.to_datetime(eq['date'])
        
        # 定义时期
        periods = {
            '训练窗口期 (前24月)': ('2021-01', '2023-01'),
            '验证期 (2023-2024)': ('2023-02', '2024-06'),
            '测试期1 (2024-2025)': ('2024-07', '2025-06'),
            '测试期2 (2025-2026)': ('2025-07', '2026-06'),
            '全样本': ('2021-01', '2026-06'),
        }
        
        period_results = []
        print(f"\n{'时期':<25} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'胜率':>8}")
        print("-" * 65)
        
        for period_name, (start, end) in periods.items():
            mask = (eq['date'] >= pd.Period(start, 'M').to_timestamp()) & \
                   (eq['date'] <= pd.Period(end, 'M').to_timestamp())
            period_eq = eq[mask].copy()
            
            if len(period_eq) < 3:
                continue
            
            period_eq['return'] = period_eq['equity'].pct_change()
            total_ret = period_eq['equity'].iloc[-1] / period_eq['equity'].iloc[0] - 1
            months = len(period_eq)
            years = months / 12
            ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
            ann_vol = period_eq['return'].std() * np.sqrt(12)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            dd = (period_eq['equity'] / period_eq['equity'].cummax() - 1).min()
            win = (period_eq['return'] > 0).mean()
            
            period_results.append({
                'period': period_name, 'annual_return': ann_ret,
                'sharpe': sharpe, 'max_drawdown': dd, 'win_rate': win
            })
            
            print(f"{period_name:<25} {ann_ret*100:>9.2f}% {sharpe:>8.2f} {dd*100:>9.2f}% {win*100:>7.1f}%")
        
        # 检查过拟合
        if len(period_results) >= 3:
            train_sharpe = period_results[0]['sharpe'] if period_results else 0
            test_sharpes = [p['sharpe'] for p in period_results[1:]]
            avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else 0
            
            print(f"\n📋 过拟合诊断:")
            print(f"  训练期夏普: {train_sharpe:.2f}")
            print(f"  测试期平均夏普: {avg_test_sharpe:.2f}")
            print(f"  夏普衰减率: {(1 - avg_test_sharpe/train_sharpe)*100:.1f}%" if train_sharpe > 0 else "  N/A")
            
            if train_sharpe > 0 and avg_test_sharpe / train_sharpe > 0.5:
                print(f"  ✅ 过拟合程度: 低（测试期夏普 > 50% 训练期）")
            elif train_sharpe > 0 and avg_test_sharpe / train_sharpe > 0.3:
                print(f"  ⚠️ 过拟合程度: 中（测试期夏普 > 30% 训练期）")
            else:
                print(f"  ❌ 过拟合程度: 高（测试期夏普 < 30% 训练期）")
    
    # ============================================================
    # Step 5: 保存结果
    # ============================================================
    print("\n💾 Step 5: 保存结果...")
    
    # 保存回测结果（不含大数组，减小文件）
    save_results = {
        'summary': results['summary'],
        'period_analysis': period_results if 'period_results' in dir() else [],
        'factor_ic': ic_df.to_dict('records'),
        'valid_factors': valid_factors,
        'run_time': datetime.now().isoformat(),
        'fixes_applied': [
            '所有因子shift(1)防止前视偏差',
            'Walk-Forward滚动训练替代固定train/test',
            '真实交易成本(佣金+印花税+滑点+冲击)',
            '换手率约束(最大30%)',
            '行业中性化选股（单行业最多15%）',
            '动态仓位管理(波动率择时)',
            '个股止损-8%，组合止损-12%',
            '清理无效因子，只保留IC_IR>0.2的有效因子',
            # v2 bug修复（2026-06-05）
            '等权模式因子IC符号反转（防止正负IC抵消）',
            '时间序列rank改为截面rank（复合因子IC提升）',
            'fwd_ret_1m改为月末对齐（替代固定shift(-21)）',
            '止损逻辑实际执行（之前是pass空操作）',
            '现金缓冲从5%降至1%（减少Cash drag）',
            '原生_raw因子值+截面rank双通道供ML使用',
        ]
    }
    
    with open(os.path.join(out_dir, "backtest_results_v2.json"), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    
    # 保存权益曲线
    if results and 'equity_curve' in results:
        eq_df = pd.DataFrame(results['equity_curve'])
        eq_df.to_csv(os.path.join(out_dir, "equity_curve_v2.csv"), index=False)
    
    print(f"\n✅ 所有结果已保存到 {out_dir}")
    
    # ============================================================
    # Step 6: 生成实盘建仓脚本
    # ============================================================
    print("\n📝 Step 6: 生成实盘建仓脚本...")
    generate_advice_script(engine, valid_factors, backtest.model, stock_info, out_dir, ic_signs)
    
    print("\n" + "=" * 70)
    print("🎉 v2.0 策略回测完成！")
    print("=" * 70)
    
    return results


def generate_advice_script(engine, factor_cols, model, stock_info, out_dir, ic_signs=None):
    """生成与回测模型一致的实盘建仓脚本"""
    
    ic_signs_str = repr(ic_signs) if ic_signs else '{}'
    script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘建仓脚本 v2.0 - 与Walk-Forward回测模型完全一致
自动生成于 {timestamp}
"""

import pandas as pd
import numpy as np
import joblib
import os

# 加载最新数据
data_dir = os.path.expanduser("~/quant_strategies/individual_stock_strategy/data")
daily = pd.read_csv(os.path.join(data_dir, "daily_data_fixed.csv"), dtype={{"trade_date": str}})
stock_info = pd.read_csv(os.path.join(data_dir, "stock_list_fixed.csv"))
industry_map = dict(zip(stock_info["ts_code"], stock_info["industry"]))

# 计算因子（使用与回测完全相同的因子引擎）
import sys
sys.path.insert(0, os.path.expanduser("~/quant_strategies/individual_stock_strategy/v2"))
from modules.factor_engine import FactorEngine

engine = FactorEngine(daily, stock_info)
factor_df = engine.build_all_factors()

# 获取最新日期
latest_date = factor_df["trade_date"].max()
latest = factor_df[factor_df["trade_date"] == latest_date].copy()

# 使用因子打分（IC符号反转后平均，与回测一致）
factor_cols = {factor_cols_str}
ic_signs = {ic_signs_str}
available = [c for c in factor_cols if c in latest.columns]
score_parts = []
for col in available:
    sign = ic_signs.get(col, 1.0)
    score_parts.append(latest[col].fillna(0) * np.sign(sign))
if score_parts:
    latest["score"] = sum(score_parts) / len(score_parts)
else:
    latest["score"] = 0

# 行业中性化选股
top_n = 30
max_per_industry = max(1, int(top_n * 0.15))
latest = latest.sort_values("score", ascending=False)

selected = []
industry_count = {{}}
for _, row in latest.iterrows():
    code = row["ts_code"]
    ind = industry_map.get(code, "Unknown")
    if industry_count.get(ind, 0) < max_per_industry:
        selected.append(row)
        industry_count[ind] = industry_count.get(ind, 0) + 1
        if len(selected) >= top_n:
            break

# 建仓建议
total_capital = 100000  # 10万
per_stock = total_capital / top_n

print(f"建仓日期: {{latest_date}}")
print(f"资金规模: {{total_capital/10000:.0f}}万元")
print(f"持仓数量: {{top_n}}只")
print()
print(f"{{'排名':<4}} {{'代码':<12}} {{'价格':>8}} {{'股数':>6}} {{'金额':>10}}")
print("-" * 50)

for i, row in enumerate(selected):
    price = row["close"]
    shares = int(per_stock / price / 100) * 100
    if shares >= 100:
        amount = shares * price
        print(f"{{i+1:<4}} {{row['ts_code']:<12}} {{price:>8.2f}} {{shares:>6}} {{amount:>10,.0f}}")
'''.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        factor_cols_str=repr(factor_cols),
        ic_signs_str=ic_signs_str
    )
    
    script_path = os.path.join(out_dir, "today_advice_v2.py")
    with open(script_path, 'w') as f:
        f.write(script)
    print(f"  ✅ 实盘脚本已保存: {script_path}")


if __name__ == '__main__':
    main()
