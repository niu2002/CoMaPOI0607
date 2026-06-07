# AMD 多 Agent 实施设计 v1

## 1. 先给结论

### 1.1 这张 AMD 卡够不够

如果采用下面这种方式，这张 AMD 卡的显存是够的：

1. 三个 Agent 分别训练，不并行训练。
2. 推理时做三 Agent 编排，但尽量只加载一份 base model。
3. LoRA adapter 以“多 adapter 共享同一基座”的方式挂载。

我的判断是：

1. **训练显存够**
   - 我们已经验证过 8B 基座在 CA 全量训练时，`BATCH_SIZE=24` 峰值约 `122.5 GiB`
   - 三个 Agent 分别训练时，不会三份同时占显存
   - 所以训练阶段只要串行跑，显存没有问题
2. **推理显存也够**
   - 如果推理采用“1 个 base model + 3 个 LoRA adapter”的服务模式，显存压力接近“1 个 8B 模型 + 少量 LoRA 额外开销”
   - 这对 192G 显存是宽裕的
3. **真正的不确定点不在显存，而在服务栈**
   - `vLLM + LoRA + ROCm + AgentScope` 这一组合能不能稳定跑，是主要风险

所以结论是：

1. **从显存角度，够。**
2. **从工程稳定性角度，需要准备一个主方案和一个后备方案。**

## 2. 关于 embedding：需不需要 API

### 2.1 最推荐方案

最推荐的是：

1. **不用远程 API**
2. 直接在本地 / AMD 机器上加载一个真实 embedding 模型
3. 生成：
   - POI embedding
   - query embedding
   - candidate cache

这样做的好处：

1. 不依赖外部 API 稳定性
2. 没有持续费用
3. 更适合反复实验

### 2.2 如果本地 embedding 模型不好用

如果本地 embedding 模型在 AMD 上不好装、速度太慢，或者 BCE 模型拿不到，我们可以退一步：

1. 用远程 embedding API 做一次性构建
2. 把结果缓存下来

要缓存的文件包括：

1. `dataset_all/{dataset}/poi_faiss_index.bin`
2. `dataset_all/{dataset}/{mode}/{dataset}_{mode}_candidates.jsonl`

这样即使使用 API，也不应该每次都消耗。

### 2.3 最终结论

embedding **不一定需要 API**。

我的建议优先级是：

1. **优先本地 embedding 模型**
2. 如果本地方案不稳定，再用 API
3. 无论走哪条路，都必须做缓存

## 3. 目标路线选择

这次实施建议走“论文式多 Agent 主路线”，但要带后备方案。

### 主路线

1. 真实 embedding + RAG candidate cache
2. 三个 Agent 分别训练
3. 多 Agent 编排推理
4. 优先尝试“1 个 base model 服务 + 3 个 LoRA adapter”

### 后备路线

如果 `vLLM + ROCm + 多 LoRA adapter` 不稳定，就退到：

1. 仍然训练三个 Agent
2. 但推理时不用服务化多 adapter
3. 改成 Python 本地顺序调用：
   - 先加载 agent1 adapter 跑
   - 再加载 agent2 adapter 跑
   - 再加载 agent3 adapter 跑

这个后备方案速度慢，但成功率更高。

## 4. 具体实施分阶段

## 阶段 1：统一训练 / 推理格式

### 目标

修掉当前最明显的质量问题：

1. 训练时是 `Question/Answer` 扁平文本
2. 推理时是 chat template + JSON 前缀
3. 两边格式不一致

### 需要修改的文件

1. `finetune_sft_new.py`
2. `lora_inference_smoke.py`
3. `lora_batch_inference_eval.py`
4. 建议新增一个公共格式化文件，比如：
   - `formatting_utils.py`

### 实施方式

1. 把训练数据改成按 message 结构序列化
2. 使用和推理一致的 tokenizer chat template
3. 保留统一的 JSON 输出约束

### 难度

中等

### 成功率

高，约 `85%~90%`

### 是否应该先做

应该，这一步优先级最高。

## 阶段 2：把 RAG embedding 改成真实版本，并加缓存

### 目标

把当前 `rag/RAG.py` 从占位实现，改成能真正提供候选 POI 的版本。

### 需要修改的文件

1. `rag/RAG.py`
2. 可能新增：
   - `ops/build_candidates.sh`
   - `ops/build_faiss_index.sh`

### 实施方式

1. 选定真实 embedding 后端
   - 首选本地 embedding 模型
   - 备选 API embedding
2. 构建 POI embedding
3. 建立 FAISS index
4. 对 train/test 生成 candidate cache
5. 所有后续实验直接复用缓存

### 决策点

需要先决定 embedding 走哪条：

1. 本地模型
2. 远程 API

### 难度

中等

### 成功率

中高，约 `75%~85%`

## 阶段 3：三 Agent 分别训练

### 目标

分别训练：

1. agent1
2. agent2
3. agent3

### 当前代码基础

当前 `finetune_sft_new.py` 已经支持：

1. `--type agent1`
2. `--type agent2`
3. `--type agent3`
4. `--type merged`

所以训练入口本身不需要推倒重来。

### 需要修改的文件

1. `ops/run_amd.sh`
2. 可能新增：
   - `ops/run_amd_agent.sh`
   - `ops/train_agent1.sh`
   - `ops/train_agent2.sh`
   - `ops/train_agent3.sh`

### 实施方式

建议把 `run_amd.sh` 扩展成支持：

1. `AGENT_TYPE=agent1`
2. `AGENT_TYPE=agent2`
3. `AGENT_TYPE=agent3`

并且分别设置输出目录，例如：

1. `finetune/results/amd-agent1/...`
2. `finetune/results/amd-agent2/...`
3. `finetune/results/amd-agent3/...`

### 显存判断

这一步显存没问题，因为是串行训练。

### 难度

中等

### 成功率

高

## 阶段 4：多 Agent 编排推理

### 目标

让 `inference_forward_new.py` 真正跑起来。

### 主方案：单服务多 LoRA

参考仓库文档的设计，优先尝试：

1. 一个 base model 服务
2. 同时挂载三个 LoRA adapter
3. 通过 `agent1` / `agent2` / `agent3` 三个名字调用

这个方案的优点是：

1. 显存最省
2. 结构最接近仓库原设计
3. 不需要三个完整基座副本

### 风险点

最大的风险是：

1. `vLLM` 在 ROCm 上是否稳定
2. `enable-lora` 在 ROCm 路线是否稳定
3. AgentScope 与这个服务是否能无缝对接

### 后备方案：本地顺序编排

如果上面的服务方案不稳定，就改成：

1. 不依赖三 Agent API 服务
2. 在 Python 脚本里顺序加载和调用三个 adapter

流程变成：

1. 先跑 agent1，拿到 profile 和候选
2. 再跑 agent2，拿到短期模式和 refined candidate
3. 再跑 agent3，做最终预测

优点：

1. 最稳
2. 对 AMD 更友好

缺点：

1. 速度慢
2. 不如服务化优雅

### 需要修改的文件

主方案可能改：

1. `inference_forward_new.py`
2. `agents.py`
3. `ops/serve_agents_amd.sh`
4. `ops/run_forward_amd.sh`

后备方案可能改：

1. `inference_forward_new.py`
2. 或新增一个：
   - `inference_forward_local_lora.py`

### 难度

主方案：高  
后备方案：中高

### 成功率

1. 主方案：中等
2. 后备方案：中高

## 5. 推荐实施顺序

建议按下面顺序做：

1. **先修训练 / 推理格式一致性**
2. **再做真实 embedding + candidate cache**
3. **再做三 Agent 分别训练**
4. **最后做多 Agent 编排推理**
5. **优先尝试主方案，失败时切后备方案**

原因很简单：

1. 前两步风险小，但收益大
2. 三 Agent 训练显存没问题
3. 最不稳定的是服务化编排，应该放到最后

## 6. 文件级修改计划

### 必改文件

1. `finetune_sft_new.py`
   - 统一训练格式
   - 支持 agent1/2/3 更明确的训练路径
2. `lora_inference_smoke.py`
   - 保持与训练格式一致
3. `lora_batch_inference_eval.py`
   - 保持与训练格式一致
4. `rag/RAG.py`
   - 替换占位 embedding
   - 增加缓存判断
5. `ops/run_amd.sh`
   - 支持按 agent 类型训练

### 高概率新增文件

1. `formatting_utils.py`
   - 统一 train / infer 输入输出格式
2. `ops/build_candidates.sh`
3. `ops/serve_agents_amd.sh`
4. `ops/run_forward_amd.sh`
5. 视情况新增：
   - `inference_forward_local_lora.py`

## 7. 实施难度与成功率评估

| 阶段 | 内容 | 难度 | 成功率 | 备注 |
|---|---|---:|---:|---|
| 阶段 1 | 修训练 / 推理格式一致性 | 中 | 85%-90% | 最优先 |
| 阶段 2 | 真实 embedding + candidate cache | 中 | 75%-85% | 候选质量关键 |
| 阶段 3 | 三 Agent 分别训练 | 中 | 85%+ | 显存足够 |
| 阶段 4A | 主方案：单服务多 LoRA 编排 | 高 | 45%-65% | 风险在 vLLM/ROCm |
| 阶段 4B | 后备方案：本地顺序编排 | 中高 | 70%-85% | 更稳但更慢 |

## 8. 最终建议

如果你的目标是：

1. “这张 AMD 卡上能真正把论文式多 Agent 做出来”

那我的建议是：

1. 这张卡 **显存上够**
2. embedding **优先本地，不强制 API**
3. 实施时采用：
   - 先修格式
   - 再做真实 embedding + cache
   - 再训练三个 Agent
   - 最后编排推理
4. 推理编排必须准备主方案和后备方案

也就是说：

1. **问题不在显存**
2. **问题在工程编排**

所以这件事值得做，而且可以做，但需要按阶段推进，不建议一口气全改。

## 9. 下一步准备怎么做

如果你确认这份实施文档，我们下一步就进入代码实施，建议顺序是：

1. 先做阶段 1：训练 / 推理格式统一
2. 同时明确 embedding 方案：
   - 本地模型优先
   - API 作为备选
3. 然后进入阶段 2 和阶段 3

这是当前最稳的开工顺序。
