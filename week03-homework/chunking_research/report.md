## 作业一: 探索 LlamaIndex 中的句子切片检索及其参数影响分析
### 作业目标
1. 理解 LlamaIndex 框架中“句子切片”的核心思想与实现机制。
2. 实践使用 LlamaIndex 构建基于句子窗口的检索增强生成（RAG）系统。
3. 对比分析不同参数设置对检索效果和生成质量的影响。
4. 培养对文本分块策略与上下文保留之间权衡的理解。

### 技术背景简介
句子切片是一种特殊的文本分块策略：将文档按**句子级别**切分，但在检索时返回包含匹配句子的“上下文窗口”，从而在精确匹配与上下文完整性之间取得平衡。

### 个人作业

#### 准备数据集：
我选取了《一句话顶一万句》、《百年孤独》、《红楼梦》、《面纱》四部小说的前4000-4500字作为数据集，每1个小说文本均采取docx形式。4个docx组成了data目录。

#### 核心代码分析：
**build_window_query_engine:**
 1. ```python
  parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,   #前后各带几句上下文
        window_metadata_key="window",  #把上下文窗口内容存在metadata的window里
        original_text_metadata_key="original_text" #把原来的单个句子存在metadata的original_text里
    )
    ```
 2. ```python
  postprocessors = [
    SimilarityPostprocessor(similarity_cutoff=similarity_cutoff), #过滤掉相似度太低的片段
    MetadataReplacementPostProcessor(target_metadata_key="window") #把检索到的句子替换成带上下文的窗口文本
    ]
    ```
 3. 功能描述：文档 → 句子窗口切片 → 生成向量索引 → 检索TOP-K → **过滤低相似度 → 还原上下文窗口** → 生成查询引擎

### chunking_research.py工作流程
1. 初始化模型（LLM + Embedding）
2. 加载小说文档
3. 普通句子分块实验 → 检索 → 回答
4. 句子窗口分块实验 → 检索 → 取出上下文 → 回答
5. 多参数句子窗口实验（6组参数 × 4个问题）
6. 输出实验结果表格
7. 保存结果到CSV文件

### CSV结果中核心参数分析
| 参数类型    | 具体参数                                      | 影响机制                                                                                | 影响程度  |
|---------| --------------------------------------------- |-------------------------------------------------------------------------------------|-------|
| 分块策略参数  | 分块器类型（Sentence vs Sentence Window）     | 决定是否保留完整语义：<br>- 普通 Sentence 分块：硬切分导致语义断裂，无法答题<br>- 句子窗口分块：按句子切分 + 上下文窗口，答题准确       | ★★★★★ |
| 检索过滤参数  | similarity_cutoff（相似度阈值）               | 决定是否过滤有效信息：<br>- cutoff≥0.5：文学文本相似度低，全部过滤，检索数 = 0<br>- cutoff=0.3：保留有效信息，检索数≥3，可正常答题 | ★★★★★ |
| 上下文窗口参数 | window_size（1/3）                            | 仅微调上下文范围，在 cutoff 合理时，值为1或者3均能答题，差异极小                                               | ★★☆☆☆ |
| 分快重叠参数  | chunk_overlap（50/100/300）                    | 仅改变文本重复度，不解决语义断裂 / 检索失效问题，所有配置均无法答题                                                 | ★☆☆☆☆ |
| 召回参数    | similarity_top_k（5/10）                      | 仅增加召回数量，但若 cutoff 过高，召回的仍是无关内容，无法答题                                                 | ★☆☆☆☆ |
