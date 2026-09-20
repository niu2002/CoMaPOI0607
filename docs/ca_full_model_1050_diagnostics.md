# 📊 CoMaPOI 多维度学术评测与空间误差分析报告
- **评测数据文件**: `results/ca/amd_rag_lora_with_sft_ca/interim_poi_predictions_1050.json`
- **总评估样本量 (Total Samples)**: `1050`

## 1. 经典推荐指标 (Recommendation Metrics)
| 指标名 | HR (命中率) | NDCG (归一化折现累计增益) |
| :--- | :---: | :---: |
| **@1** | 12.57% | 12.57% |
| **@3** | 27.33% | 21.07% |
| **@5** | 33.71% | 23.69% |
| **@10** | 42.48% | 26.53% |
| **MRR** | **21.54%** | - |

## 2. 空间合理性分析 (Spatial Feasibility Analysis)
| 空间评估维度 | 指标数值 | 物理意义 |
| :--- | :---: | :--- |
| **Mean Distance Error (MDE)** | **38.417 km** | Top-1 预测 POI 距离真实 POI 的平均大圆距离 |
| **Median Distance Error** | **5.889 km** | Top-1 预测 POI 距离真实 POI 的中位数距离 |
| **Spatial Recall @1.0km** | 30.76% | Top-1 预测在真实 POI 地理半径 1 公里内的比例 |
| **Spatial Recall @3.0km** | 39.81% | Top-1 预测在真实 POI 地理半径 3 公里内的比例 |
| **Spatial Recall @5.0km** | 47.24% | Top-1 预测在真实 POI 地理半径 5 公里内的比例 |

## 3. 指令对齐与合规性诊断 (Alignment & Feasibility Diagnostics)
| 诊断维度 | 异常频次 / 总数 | 百分比 | 物理说明 |
| :--- | :---: | :---: | :--- |
| **格式解析错误 (Format Error)** | 0 / 1050 | 0.00% | 大模型未能生成合规 JSON 列表而导致解析为空 |
| **预测长度溢出 (Length Mismatch)** | 16 / 1050 | 1.52% | 推荐列表的推荐项不等于 10 个 |
| **越界推荐率 (Out-of-Candidates)** | 0 / 10479 | 0.00% | 推荐的 POI 不在多源融合召回候选池中 (幻觉) |