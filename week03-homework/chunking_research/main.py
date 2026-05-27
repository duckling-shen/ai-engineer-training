import os
import pandas as pd

# 环境变量配置
os.environ["DEEPSEEK_API_KEY"] = "sk-xxxxxxxxxx"
#此处使用的是DEEPSEEK配置的API

# 核心依赖导入
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 切片模块
from llama_index.core.node_parser import (
    SentenceSplitter,
    SentenceWindowNodeParser
)
# 后处理器
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    MetadataReplacementPostProcessor
)
# 检索与查询引擎
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# ===================== 全局模型配置 =====================
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    is_chat_model=True,
    temperature=0.1
)

# 本地向量模型配置
Settings.embed_model = HuggingFaceEmbedding(
    model_name="/mnt/HDD/models/bce-embedding-base-v1"
)

# ===================== 核心函数定义 =====================
def evaluate_splitter(
        splitter, documents, query: str, ground_truth: str,
        splitter_name: str, chunk_overlap=0
):
    """分块效果评估函数"""
    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    # 上下文冗余程度评分（1=无冗余，5=严重冗余）
    if isinstance(splitter, SentenceWindowNodeParser):
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")]
        )
        context_redundancy = 2  # 句子窗口：中等冗余
    else:
        query_engine = index.as_query_engine(similarity_top_k=5, similarity_cutoff=0.6)
        # 根据 overlap 大小判定冗余
        if chunk_overlap <= 50:
            context_redundancy = 1  # 小重叠：极低冗余
        elif chunk_overlap <= 100:
            context_redundancy = 3  # 中重叠：低冗余
        else:
            context_redundancy = 4  # 大重叠：高冗余

    response = query_engine.query(query)
    retrieved_num = len(response.source_nodes)

    # 打印实验结果
    print(f"\n===== 分块方式：{splitter_name} | overlap={chunk_overlap} =====")
    print(f"问题：{query}")
    print(f"检索节点数：{retrieved_num}")
    print(f"上下文冗余评分(1-5)：{context_redundancy}")
    print(f"生成回答：{str(response)[:300]}")
    print(f"标准答案：{ground_truth}")
    print("=" * 70)

    return {
        "splitter": splitter_name,
        "chunk_overlap": chunk_overlap,
        "query": query,
        "retrieved_nodes": retrieved_num,
        "context_redundancy": context_redundancy,
        "answer": str(response)[:300]
    }


def build_window_query_engine(
        documents, window_size=1, similarity_top_k=5, similarity_cutoff=0.7
):
    """构建句子窗口查询引擎"""
    parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
    postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=similarity_cutoff),
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ]

    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=postprocessors
    )


# ===================== 主函数 =====================
def main():
    print("========== 开始执行分块对比实验 ==========\n")

    # Step1: 加载文档
    documents = SimpleDirectoryReader(
        input_dir="data", required_exts=[".docx"], recursive=False
    ).load_data()
    print(f"已成功加载 {len(documents)} 个文档！")

    test_question = "老杨和老马真实关系如何？"
    test_ground_truth = "老杨真心把老马当知心朋友、凡事愿意交心；但老马内心看不起老杨，只是拿他消遣，并不把老杨当作真正朋友，属于表面交好、内心不对等的单向友谊。"

    # ===================== 实验1：chunk_overlap 大小对比 =====================
    print("\n========== 实验1：chunk_overlap 大小对效果的影响 ==========")
    overlap_configs = [50, 100, 150]  # 小、中、大 三种重叠尺寸
    overlap_results = []
    for overlap in overlap_configs:
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=overlap)
        res = evaluate_splitter(
            splitter, documents, test_question, test_ground_truth,
            f"Sentence_overlap_{overlap}", overlap
        )
        overlap_results.append(res)

    # ===================== 实验2：普通切片 vs 句子窗口切片对比 =====================
    print("\n========== 实验2：分块方式效果对比 ==========")
    # 普通句子切片
    sentence_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    evaluate_splitter(sentence_splitter, documents, test_question, test_ground_truth, "Sentence", 50)

    # 句子窗口切片
    sentence_window_splitter = SentenceWindowNodeParser.from_defaults(
        window_size=4, window_metadata_key="window", original_text_metadata_key="original_text"
    )
    evaluate_splitter(sentence_window_splitter, documents, test_question, test_ground_truth, "Sentence Window", 0)

    # ===================== 实验3：句子窗口参数对比实验（控制变量） =====================
    print("\n========== 实验3：句子窗口参数对比 ==========")
    test_queries = [
        "《红楼梦》开篇空空道人为什么改名情僧？",
        "面纱中沃尔特给凯蒂哪两个选择？",
        "百年孤独里布恩迪亚认为地球是什么形状？",
        "一句话顶一万句中老杨和老马关系如何？"
    ]

    # 控制变量法设计参数（对比window_size/top_k/cutoff）
    test_configs = [
        {"window_size": 1, "similarity_top_k": 5, "similarity_cutoff": 0.5},
        {"window_size": 3, "similarity_top_k": 5, "similarity_cutoff": 0.5},
        {"window_size": 1, "similarity_top_k": 10, "similarity_cutoff": 0.5},
        {"window_size": 1, "similarity_top_k": 5, "similarity_cutoff": 0.3},
        {"window_size": 3, "similarity_top_k": 5, "similarity_cutoff": 0.3},
        {"window_size": 3, "similarity_top_k": 10, "similarity_cutoff": 0.3}
    ]

    experiment_results = []
    for idx, config in enumerate(test_configs):
        print(f"\n=== 测试配置 {idx + 1}: {config} ===")
        engine = build_window_query_engine(documents, **config)
        for q in test_queries:
            res = engine.query(q)
            # 主观冗余评分：窗口越大/top_k越大，冗余越高
            redundancy = 2 if config["window_size"] == 1 else 4
            experiment_results.append({
                "config": idx + 1,
                "window_size": config["window_size"],
                "top_k": config["similarity_top_k"],
                "similarity_cutoff": config["similarity_cutoff"],
                "query": q,
                "retrieved_nodes": len(res.source_nodes),
                "context_redundancy": redundancy,
                "answer": str(res)[:200]
            })

    # ===================== 结果保存与输出 =====================
    # 保存所有实验结果
    df = pd.DataFrame(experiment_results)
    df_overlap = pd.DataFrame(overlap_results)

    print("\n========== 最终实验结果 ==========")
    print("\n--- chunk_overlap 对比结果 ---")
    print(df_overlap[["splitter", "chunk_overlap", "retrieved_nodes", "context_redundancy"]])

    print("\n--- 句子窗口参数对比结果 ---")
    print(df[["config", "window_size", "top_k", "similarity_cutoff", "retrieved_nodes", "context_redundancy"]])

    df.to_csv("result.csv", index=False, encoding="utf-8")
    df_overlap.to_csv("chunk_overlap_analysis.csv", index=False, encoding="utf-8")

    print("\n结果已保存至 CSV 文件")
    print("========== 实验执行完成 ==========")


# ===================== 程序入口 =====================
if __name__ == "__main__":
    main()