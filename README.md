# A股个股多因子量化策略工作流

## 📊 项目概述

本项目是一个完整的A股个股多因子量化策略研究框架，包含因子挖掘、策略构建、回测验证等全流程。

### 核心成果

| 策略 | 年化收益 | 最大回撤 | 夏普比率 | Calmar比率 | 胜率 |
|------|---------|---------|---------|-----------|------|
| **ML+LLM等权组合** | **46.60%** | -19.00% | **1.56** | **2.45** | **67.2%** |
| ML因子策略 | 38.73% | **-9.92%** | **8.19** | **3.90** | 65.2% |
| LLM单因子策略 | 29.32% | -28.11% | 1.10 | 1.04 | 63.5% |

## 🏗️ 项目结构

```
a-stock-quant-workflow/
├── README.md                          # 项目说明
├── scripts/                           # 策略脚本
│   ├── llm_factor_part1.py           # LLM因子构建+IC分析
│   └── llm_factor_part2.py           # LLM回测+ML+LLM组合
├── data/                              # 数据文件
│   ├── daily_data_fixed.csv          # 修复后的日线数据(515只股票)
│   ├── factor_data.csv               # 因子数据
│   ├── ml_factor_scores.csv          # ML因子得分
│   ├── ml_feature_importance.csv     # ML特征重要性
│   ├── llm_factors.csv               # 13个LLM因子数据
│   ├── llm_factor_ic.csv             # LLM因子IC分析
│   ├── llm_ic_results.json           # IC分析结果
│   ├── llm_backtest_results.json     # LLM回测结果
│   └── backtest_result_ml.json       # ML回测结果
└── reports/                           # 研究报告
    └── research_report_fixed.md      # 详细研究报告
```

## 🔬 因子体系

### ML因子 (15个)
- 技术指标: MA比率、布林带宽度、波动率
- 动量因子: 5日/20日/60日动量
- 流动性: Amihud非流动性、日内波动率
- 价格位置: 20日高低点位置

### LLM因子 (13个)
- 情绪因子: 恐慌指数、贪婪指数、情绪一致性
- 动量情绪: 多周期动量综合、反转情绪
- 微观结构: 量价背离、信息不对称、流动性情绪
- 事件驱动: 关注度因子、事件驱动因子
- 复合因子: 复合情绪、LLM情绪得分

### 因子有效性 (IC_IR > 0.3)
| 因子 | IC | IC_IR | 有效性 |
|------|-----|-------|--------|
| liquidity_sentiment | -0.0576 | -0.5135 | ✅ 反向有效 |
| ML因子(GRM) | 0.0971 | 0.9167 | ✅ 正向有效 |

## 📈 策略详情

### 1. ML因子策略 (最佳风险调整)
- **方法**: GradientBoosting + 时间序列交叉验证
- **选股**: Top 50等权
- **调仓**: 月度
- **特点**: 夏普8.19，回撤仅-9.92%

### 2. ML+LLM等权组合 (最高收益)
- **方法**: ML得分(70%) + LLM流动性情绪(30%)
- **选股**: Top 50等权
- **调仓**: 月度
- **特点**: 年化46.60%，胜率67.2%

### 3. LLM单因子策略
- **因子**: liquidity_sentiment (反向)
- **选股**: 流动性最好Top 50
- **调仓**: 月度
- **特点**: 年化29.32%

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/guet-tyx/a-stock-quant-workflow.git
cd a-stock-quant-workflow

# 2. 安装依赖
pip install pandas numpy scikit-learn tushare

# 3. 运行LLM因子分析
python scripts/llm_factor_part1.py

# 4. 运行回测
python scripts/llm_factor_part2.py
```

## 📊 数据说明

- **数据源**: Tushare
- **股票池**: 沪深300+中证500成分股 (515只)
- **时间范围**: 2021-01-04 ~ 2026-06-04
- **频率**: 日线数据，月度调仓

## ⚠️ 风险提示

1. 本项目仅供研究学习，不构成投资建议
2. 历史回测不代表未来表现
3. 实盘交易需考虑滑点、手续费、冲击成本
4. 因子可能存在周期性失效风险

## 📝 更新日志

### v1.0 (2026-06-05)
- 完成ML因子挖掘与回测
- 完成LLM因子构建与验证
- 实现ML+LLM组合策略
- 最优策略年化46.60%，夏普1.56

## 👤 作者

- GitHub: [guet-tyx](https://github.com/guet-tyx)

## 📄 License

MIT License
