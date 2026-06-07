# NYC Full-Flow Handoff

把下面这段直接贴给 AI，然后把真实输出路径补进去。

```text
我刚在本机跑完 CoMaPOI0607 的 NYC 全流程，请继续分析结果。

项目信息:
- 仓库: F:\jjy\project\CoMaPoi0607
- 环境: co0607
- 基础模型: F:\jjy\project\CoMaPoi0607\models\Qwen3.5-2B
- 数据集: nyc
- 训练集大小: 3870
- 测试集大小: 988
- 训练方式: LoRA, batch_size=2, gradient_accumulation_steps=1, max_steps=100
- 数据语义: --num_samples 0 表示全量样本

训练命令输出目录:
- 权重目录: F:\jjy\project\CoMaPoi0607\finetune\results\6-07-full\sft-nyc\bs2-gas1-ms100-merged-lr0.0001
- 训练日志: {{把 train 日志路径填到这里}}

评估命令输出目录:
- 评估目录: {{把 artifacts\eval\nyc\lora_eval_时间戳 目录填到这里}}
- 评估日志: {{把 infer_eval 日志路径填到这里}}
- metrics.txt: {{把 metrics.txt 路径填到这里}}
- summary.json: {{把 summary.json 路径填到这里}}

请你做这些事:
1. 读取训练日志，判断训练是否正常结束，有没有 OOM、编码问题或异常退出。
2. 读取 metrics.txt 和 summary.json，总结 HR@1/5、MRR、NDCG@5，以及总耗时和显存占用。
3. 如果目录里有 predictions.json，请进一步分析命中分布，区分 top1 命中和 top5 命中。
4. 和之前 NYC 50-sample 基线比较。之前基线是 HR@1=32.00, HR@5=44.00, MRR=37.33, NDCG@5=39.05。
5. 给出下一步建议: 是继续拉长 step，还是转去 agent1/agent2/agent3 的单独微调。
```
