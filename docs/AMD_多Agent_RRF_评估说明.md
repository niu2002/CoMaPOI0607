# AMD 分支多 Agent / RRF 评估说明

## 1. 这份文档是干什么的

这份文档是为了回答三个问题：

1. 在当前 `amd` 分支下，如果要往论文里的多 Agent CoMaPOI 靠近，具体要做什么。
2. 现在训练格式和推理格式不一致，应该怎么改。
3. 为什么当前已经跑通的 `merged LoRA` 路线，并不等于论文里的 RRF 设定。

这份文档先做“评估和决策”，不直接改代码。

## 2. 当前 amd 分支已经做到什么程度了

当前 `amd` 分支已经证明了下面几件事：

1. 在 AMD ROCm 上，`Llama-3.1-8B-Instruct` 可以完成 CA 全量 LoRA 训练。
2. LoRA 的 `smoke` 和 `eval` 都能完整跑通。
3. 训练中断后的 checkpoint / resume 机制已经补上了。

但是，现在跑通的是一条“简化版流程”：

1. 训练时用的是 `finetune_sft_new.py --type merged`
2. 推理时用的是 `lora_inference_smoke.py` / `lora_batch_inference_eval.py`
3. 本质上是在做“给定轨迹，直接生成 `next_poi_id`”

这条路线是一个可运行的工程基线，但它还不是论文主方法的完整复现。

## 3. 论文里的 CoMaPOI 真正需要什么

论文式 CoMaPOI 不是“一个模型直接预测下一个 POI”，而是一个多阶段流程：

1. 逆向推理 / RRF 数据生成
   - `inference_inverse_new.py`
   - 生成 `agent1_train_samples.jsonl`
   - 生成 `agent2_train_samples.jsonl`
   - 生成 `agent3_train_samples.jsonl`
2. 分别微调三个 Agent
   - `finetune_sft_new.py --type agent1`
   - `finetune_sft_new.py --type agent2`
   - `finetune_sft_new.py --type agent3`
3. 多 Agent 前向推理
   - `inference_forward_new.py`
   - Profiler -> Forecaster -> Final_Predictor
4. 候选集收缩
   - `rag/RAG.py`
   - 先生成候选 POI，再把候选集合交给最终预测器

所以论文最强的设定，不是“merged LoRA 直接输出 POI ID”，而是“多个角色分工协作，再输出最终答案”。

## 4. 如果在 AMD 上做多 Agent，需要补什么

### 4.1 RAG / 嵌入层

当前 `rag/RAG.py` 还不能直接当成论文级实现使用。

原因很关键：

1. 文件里看起来像是要用 `models/bce_embedding`
2. 但实际 `EmbeddingModel` 类只是一个“占位实现”
3. 它现在不是在做真正的语义嵌入，而是用文本哈希伪造 embedding

这意味着：

1. 当前 RAG 代码结构是有的
2. 但候选 POI 的语义质量并不真实
3. 所以如果不先换成真实 embedding，后面的 candidate refinement 就不可靠

### 4.2 这个文本嵌入模型是不是只需要嵌入一次

你的理解基本是对的，但要分两层：

1. POI 库 embedding
   - 应该只做一次
   - 做完保存成 FAISS index
   - 后续重复使用
2. 测试 / 训练样本的 query embedding
   - 首次生成 candidate 文件时需要做
   - 但生成完 `*_candidates.jsonl` 以后也可以缓存
   - 后续重复 eval 不需要重新走 API

所以正确的缓存策略应该是：

1. `dataset_all/{dataset}/poi_faiss_index.bin` 只建一次
2. `dataset_all/{dataset}/{mode}/{dataset}_{mode}_candidates.jsonl` 每个 split 只建一次
3. 后续多次训练、推理、评测都复用

也就是说：

1. 嵌入 API 不应该每次评测都耗费
2. 最理想是“第一次建库/建候选时消耗，之后复用缓存”

### 4.3 剩下两个 Agent 是不是也是微调的

是的，而且严格来说一共是三个微调角色：

1. Agent 1
   - 长期画像
   - 基于历史行为生成候选 POI
2. Agent 2
   - 短期移动模式
   - 结合 RAG 候选做 refinement
3. Agent 3
   - 综合前两个 Agent 的结果
   - 预测最终 `next_poi_id`

仓库代码本身就是按这个结构设计的，相关文件包括：

1. `prompt_provider.py`
2. `ft_data.py`
3. `inference_forward_new.py`
4. `finetune_sft_new.py --type agent1/agent2/agent3`

## 5. 现在有两条可行路线

### 路线 A：按论文方式做 full multi-agent

目标是：

1. 准备或重建 RRF 数据
2. 分别训练 agent1 / agent2 / agent3
3. 做真实 embedding + 候选集缓存
4. 跑 `inference_forward_new.py`
5. 用多 Agent 协作方式做最终推理

这条路线最接近论文，但工作量最大。

需要解决的问题：

1. `rag/RAG.py` 要换成真实 embedding
2. `finetune/data/{dataset}` 要完整可用
3. 三个 Agent 分别训练
4. 三个 Agent 要通过 OpenAI-compatible API 被 AgentScope 调起来
5. AMD 上要能稳定支持这种服务编排

这条路线的风险最大的一点，不是训练，而是“服务和调度”：

1. AgentScope
2. 三个 Agent API
3. LoRA adapter 服务化
4. AMD / ROCm 上的兼容性

我的判断：

1. 论文还原度：高
2. 工程复杂度：高
3. 在当前 AMD 环境下的成功率：中等

粗略成功概率评估：

1. 如果保持现在这种 API / AgentScope 结构，成功率大概 `40%~55%`
2. 如果后面改成更简单的本地顺序执行版，不强依赖三路服务，成功率大概 `60%~75%`

### 路线 B：先做 RRF 对齐的单模型路线

目标是：

1. 暂时保留一个 merged 模型
2. 但不要再让它只做“直接输出 POI”
3. 而是让它更接近论文里 RRF 的单模型能力
4. 也就是显式学习中间结构和推理过程

优点：

1. 不需要三路服务编排
2. 更适合 AMD 先落地
3. 比当前“直接输出 POI”的 merged LoRA 更接近论文

我的判断：

1. 对 full CoMaPOI 的还原度：中等
2. 对论文里 RRF 单模型设定的贴近度：中高
3. 工程复杂度：中等
4. 成功率：`70%~85%`

## 6. 为什么当前 merged LoRA 不等于论文里的 RRF

这是最关键的一点。

### 6.1 当前 merged LoRA 在做什么

现在 AMD 上这条路是：

1. `finetune_sft_new.py --type merged`
2. 把 agent1 / agent2 / agent3 的训练样本混在一起训练
3. 推理时用 `lora_inference_smoke.py` / `lora_batch_inference_eval.py`
4. 直接要求模型生成 `next_poi_id`

也就是说，它现在的行为更像：

1. “给你一段轨迹，直接猜下一个 POI 是谁”

### 6.2 论文里的 RRF 是什么意思

论文里的 RRF 不是“把数据 merge 一下就行了”。

RRF 的核心是：

1. 通过逆向推理构造训练监督
2. 让模型学会中间结构
3. 这些中间结构包括：
   - 长期画像
   - 短期模式
   - 候选集优化
   - 最终预测

即使最后是一个 merged 模型，它也应该体现出比“直接吐一个 POI ID”更丰富的行为。

### 6.3 当前 merged 路线和论文 RRF 的真实差异

当前 merged 路线偏弱，主要有三个原因：

1. 训练目标是混合的，但推理目标被压缩成了“只输出最终 ID”
2. agent2 学到的 candidate refinement 信号在推理时没有被显式利用
3. 最终预测时没有真正受到 candidate set 约束，论文最重要的“缩小候选空间”优势消失了

所以当前 merged LoRA 应该被理解成：

1. 一个工程化简化基线
2. 不是论文里的 RRF
3. 更不是论文里的 full CoMaPOI

## 7. 训练格式和推理格式为什么不一致

### 7.1 当前不一致点

当前训练：

1. `prepare_sample_text()` 把样本变成纯文本
2. 形式类似：
   - `Question: ...`
   - `Answer: ...`

当前推理：

1. 用 `tokenizer.apply_chat_template(...)`
2. 再手工加上 `{"next_poi_id": ` 前缀
3. 实际是在 chat completion 风格下做生成

这两者明显不是同一种输入分布。

### 7.2 为什么这会影响效果

因为模型训练时看到的是：

1. 扁平文本

而推理时看到的是：

1. system / user / assistant 的 chat 格式

这会导致：

1. 模型学到的格式先验和推理时不匹配
2. 输出容易漂
3. 需要靠额外的 JSON 前缀约束来“拉回去”

### 7.3 建议怎么改

建议后续统一成：

1. 训练和推理都保留 message 结构
2. 训练时也走 chat template
3. 训练和推理使用同一套输出 JSON 约束

这样改的收益很大：

1. 能同时提升当前 merged 路线
2. 也能为后面多 Agent 铺路

我的判断：

1. 难度：中等
2. 收益：中高
3. 成功率：`85%~90%`

## 8. 建议的实施顺序

### 第一阶段：先修训练 / 推理格式不一致

为什么优先做这个：

1. 风险最小
2. 对当前 merged 路线立刻有帮助
3. 对后面的多 Agent 也有帮助

判断：

1. 难度：中等
2. 优先级：最高

### 第二阶段：把 embedding / RAG 候选集改成真实可缓存版本

为什么第二步做：

1. 候选集质量是 CoMaPOI 的核心
2. 当前 `rag/RAG.py` 的占位 embedding 不能支撑论文级结果

判断：

1. 难度：中等
2. 优先级：高

### 第三阶段：在路线 A 和路线 B 中二选一

我的默认建议是：

1. 如果目标是 AMD 上尽快拿到更可靠结果，优先路线 B
2. 如果目标是尽量贴近论文原方法，再走路线 A

原因：

1. 路线 B 更容易成功
2. 路线 A 更像论文，但服务编排风险很高

## 9. 我的总体建议

如果你当前目标是：

1. “先让 AMD 路线更像论文，且成功率高”

那最建议的顺序是：

1. 先修训练 / 推理格式一致性
2. 再把真实 embedding + candidate cache 接上
3. 再把 merged 路线改成更接近 RRF 的推理方式
4. 最后再决定要不要投入 full multi-agent 服务化

如果你当前目标是：

1. “尽量严格复现论文主流程”

那就需要准备做：

1. 真实 embedding
2. candidate cache
3. agent1 / agent2 / agent3 分别微调
4. 三路 Agent 服务化
5. `inference_forward_new.py` 全流程验证

这条路线可做，但明显更重、更慢、风险更高。

## 10. 难度和成功率汇总

| 项目 | 改什么 | 难度 | 成功率 | 说明 |
|---|---|---:|---:|---|
| 修训练 / 推理格式不一致 | 统一 train 和 infer 的格式 | 中 | 85%-90% | 最应该先做 |
| 替换 placeholder embedding | 真正做语义检索 | 中 | 75%-85% | 论文式候选集的前提 |
| 做 embedding / candidate 缓存 | 避免重复消耗 API | 低-中 | 90%+ | 工程上最直接 |
| 做 RRF 对齐的 merged 推理 | 单模型但更接近论文 | 中 | 70%-85% | 最适合 AMD 先落地 |
| 做 full multi-agent | 三 Agent 微调 + 服务化 + 协同推理 | 高 | 40%-75% | 最接近论文，但最重 |

## 11. 下一份文档应该写什么

如果你认可这份评估，下一份应该写“实施设计文档”，内容包括：

1. 具体改哪些文件
2. 是走路线 A 还是路线 B
3. 训练命令怎么改
4. 推理命令怎么改
5. embedding 和 candidate cache 怎么做
6. AMD 上如何做 checkpoint / 回滚 / 恢复

我建议你先确认这份评估，再进入实施设计文档阶段。
