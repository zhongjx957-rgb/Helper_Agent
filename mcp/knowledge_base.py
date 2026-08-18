"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import hashlib
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """混合检索融合策略。"""
    RRF = "rrf"               # Reciprocal Rank Fusion
    WEIGHTED = "weighted"     # 加权平均融合


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 使用服务端时不传 embedding_function，让服务端处理
        # 本地模式时也不传，使用 ChromaDB 默认的（会触发模型下载）
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "We_Listen RAG 知识库"},
        )

        # 如果知识库为空，导入默认文档
        if self._collection.count() == 0:
            self._load_default_docs()

        # ── BM25 关键词检索 相关属性 ──────────────────────────────────────────
        self._bm25_built: bool = False
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus: List[str] = []
        self._bm25_tokenized: List[List[str]] = []
        self._bm25_doc_ids: List[str] = []
        self._bm25_config: Dict[str, Any] = {
            "k1": 1.5,       # 词频饱和参数
            "b": 0.75,       # 长度归一化参数
        }
        # 自定义词典路径（可选，通过环境变量配置）
        _user_dict = os.getenv("JIEBA_USER_DICT", "")
        if _user_dict:
            jieba.load_userdict(_user_dict)
            logger.info(f"jieba 已加载自定义词典: {_user_dict}")

        # 自动构建 BM25 索引
        self._init_bm25_index()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({"title": title, "chunk_index": i, "total_chunks": len(chunks)})

        if ids:
            # ChromaDB 会自动生成 Embedding
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库导入 {len(ids)} 个文档片段")
            # 重建 BM25 索引以包含新文档
            self._rebuild_bm25_index()

        return len(ids)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 内部自动将 query 转为向量，与存储的文档向量做余弦相似度匹配。
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist, doc_id in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                    "_id":      doc_id,  # ChromaDB ID，供混合检索对齐使用
                })

        return items

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    # ── BM25 索引管理 ─────────────────────────────────────────────────────────

    def _init_bm25_index(self) -> None:
        """
        从 ChromaDB 全量读取文档块，用 jieba 分词后构建 BM25Okapi 倒排索引。
        在初始化时自动调用。
        """
        try:
            all_docs = self._collection.get()
            if not all_docs or not all_docs.get("documents"):
                self._bm25_built = True  # 空知识库也算就绪
                logger.info("BM25 索引: 知识库为空，跳过索引构建")
                return

            documents = all_docs["documents"]
            doc_ids = all_docs.get("ids", [])
            if not documents:
                self._bm25_built = True
                return

            # 分词
            tokenized = [list(jieba.lcut_for_search(doc)) for doc in documents]

            self._bm25_corpus = list(documents)
            self._bm25_tokenized = tokenized
            self._bm25_doc_ids = list(doc_ids)

            # 重建 BM25 索引
            self._bm25_index = BM25Okapi(
                tokenized,
                k1=self._bm25_config["k1"],
                b=self._bm25_config["b"],
            )
            self._bm25_built = True
            logger.info(f"BM25 索引构建完成: {len(self._bm25_corpus)} 个文档块, "
                        f"k1={self._bm25_config['k1']}, b={self._bm25_config['b']}")
        except Exception as ex:
            self._bm25_built = False
            logger.error(f"BM25 索引构建失败: {ex}")

    def _bm25_search(self, query: str) -> List[Tuple[int, float]]:
        """
        BM25 关键词检索。

        Args:
            query: 用户查询字符串

        Returns:
            List[(corpus_index, bm25_score)] — 按 BM25 得分降序排列
        """
        if not self._bm25_built or self._bm25_index is None:
            return []

        query_tokens = list(jieba.lcut_for_search(query))
        scores = self._bm25_index.get_scores(query_tokens)

        # 按得分降序排列
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )
        # 过滤掉得分为 0 的结果
        ranked = [(idx, score) for idx, score in ranked if score > 0]
        return ranked

    def _rebuild_bm25_index(self) -> None:
        """从当前 ChromaDB 全量重建 BM25 索引（用于文档增删后同步）。"""
        self._init_bm25_index()

    # ── 融合策略 ───────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        bm25_rankings: List[Tuple[int, float]],
        vector_rankings: List[Tuple[int, float]],
        top_k: int,
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF) 融合。
        将两路排位列表按 RRF 公式融合：
            RRF_score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_vector(d))

        Args:
            bm25_rankings:    BM25 排名列表 [(idx, score), ...] 按得分降序
            vector_rankings:  向量排名列表 [(idx, score), ...] 按得分降序
            top_k:            返回 Top-K
            k:                RRF 平滑常数（默认 60）

        Returns:
            List[(doc_index, fusion_score)] — 按融合得分降序
        """
        rrf_scores: Dict[int, float] = {}

        # BM25 贡献
        for rank_pos, (doc_idx, _) in enumerate(bm25_rankings):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank_pos + 1)

        # 向量贡献
        for rank_pos, (doc_idx, _) in enumerate(vector_rankings):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank_pos + 1)

        # 按融合分降序排列
        sorted_scores = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_scores[:top_k]

    @staticmethod
    def _weighted_fuse(
        bm25_rankings: List[Tuple[int, float]],
        vector_rankings: List[Tuple[int, float]],
        top_k: int,
        alpha: float = 0.5,
    ) -> List[Tuple[int, float]]:
        """
        加权平均融合。将两路得分归一化到 [0,1] 后加权求和。
        公式: fusion = alpha * vector_norm + (1-alpha) * bm25_norm

        Args:
            bm25_rankings:    BM25 排名 [(idx, score), ...]
            vector_rankings:  向量排名 [(idx, score), ...]
            top_k:            返回 Top-K
            alpha:            向量权重（0-1），BM25 权重为 1-alpha

        Returns:
            List[(doc_index, fusion_score)] — 按融合得分降序
        """
        # 构建 {doc_idx: score} 映射
        bm25_map = {idx: score for idx, score in bm25_rankings}
        vector_map = {idx: score for idx, score in vector_rankings}

        # 收集所有涉及的文档索引
        all_indices = set(bm25_map.keys()) | set(vector_map.keys())

        def _min_max_norm(scores_map: Dict[int, float]) -> Dict[int, float]:
            """Min-Max 归一化到 [0, 1]"""
            if not scores_map:
                return {}
            values = list(scores_map.values())
            min_v, max_v = min(values), max(values)
            if max_v == min_v:
                return {k: 0.5 for k in scores_map}
            return {k: (v - min_v) / (max_v - min_v) for k, v in scores_map.items()}

        bm25_norm = _min_max_norm(bm25_map)
        vector_norm = _min_max_norm(vector_map)

        fused = {}
        for idx in all_indices:
            b_score = bm25_norm.get(idx, 0.0)
            v_score = vector_norm.get(idx, 0.0)
            fused[idx] = alpha * v_score + (1.0 - alpha) * b_score

        sorted_scores = sorted(
            fused.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_scores[:top_k]

    # ── 混合检索 ───────────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        混合检索：BM25 关键词检索 + ChromaDB 向量检索 → RRF/加权融合。

        Args:
            query:    用户查询
            top_k:    返回 Top-K 结果
            strategy: 融合策略 "rrf" | "weighted"
            rrf_k:    RRF 平滑常数
            alpha:    加权融合中向量的权重（0-1）

        Returns:
            结果列表，每项包含 title, content, bm25_score, vector_score,
            fusion_score, chunk, source
        """
        if not self._bm25_built:
            logger.warning("BM25 索引未就绪，回退到纯向量检索")
            return self.search(query, top_k=top_k)

        # 1. BM25 检索
        bm25_rankings = self._bm25_search(query)

        # 2. 向量检索（多取一些候选，供融合排序使用）
        recall_k = max(top_k * 2, 10)
        vector_results = self.search(query, top_k=recall_k)
        if self._bm25_doc_ids:
            vector_rankings = []
            for item in vector_results:
                doc_id = item.get("_id", "")
                if doc_id in self._bm25_doc_ids:
                    corpus_idx = self._bm25_doc_ids.index(doc_id)
                    vector_rankings.append((corpus_idx, item.get("score", 0.0)))
        else:
            vector_rankings = [
                (i, item.get("score", 0.0))
                for i, item in enumerate(vector_results)
            ]

        # 3. 融合
        if strategy == FusionStrategy.WEIGHTED:
            fused = self._weighted_fuse(bm25_rankings, vector_rankings, top_k, alpha=alpha)
        else:
            fused = self._rrf_fuse(bm25_rankings, vector_rankings, top_k, k=rrf_k)

        # 4. 组装结果
        result_map = {}
        for idx, score in bm25_rankings:
            if idx < len(self._bm25_corpus):
                result_map[idx] = {
                    "content": self._bm25_corpus[idx],
                    "bm25_score": round(score, 4),
                    "vector_score": 0.0,
                }
        for idx, score in vector_rankings:
            if idx < len(self._bm25_corpus):
                if idx in result_map:
                    result_map[idx]["vector_score"] = round(score, 4)
                else:
                    result_map[idx] = {
                        "content": self._bm25_corpus[idx] if idx < len(self._bm25_corpus) else "",
                        "bm25_score": 0.0,
                        "vector_score": round(score, 4),
                    }

        items = []
        for idx, f_score in fused:
            if idx not in result_map:
                continue
            info = result_map[idx]
            # 尝试从 ChromaDB metadata 取 title
            title = ""
            if self._bm25_doc_ids and idx < len(self._bm25_doc_ids):
                try:
                    meta_result = self._collection.get(
                        ids=[self._bm25_doc_ids[idx]],
                        include=["metadatas"],
                    )
                    if meta_result and meta_result.get("metadatas") and meta_result["metadatas"][0]:
                        title = meta_result["metadatas"][0].get("title", "")
                except Exception:
                    pass

            items.append({
                "title":    title,
                "content":  info["content"],
                "bm25_score":  info["bm25_score"],
                "vector_score": info["vector_score"],
                "fusion_score": round(f_score, 4),
                "source":   "hybrid",
            })

        # 若融合结果不足，用向量检索结果补足
        if len(items) < top_k:
            existing_ids = {item.get("content", "") for item in items}
            for v_item in vector_results:
                if v_item.get("content", "") not in existing_ids:
                    v_item["source"] = "hybrid"
                    v_item["bm25_score"] = 0.0
                    v_item["vector_score"] = v_item.get("score", 0.0)
                    v_item["fusion_score"] = v_item.get("score", 0.0)
                    items.append(v_item)
                    existing_ids.add(v_item.get("content", ""))
                if len(items) >= top_k:
                    break

        logger.debug(f"混合检索: query={query!r}, strategy={strategy}, "
                      f"结果数={len(items)}")
        return items[:top_k]

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        # 默认走混合检索（与工具 schema 的 default=True 保持一致）
        hybrid = params.get("hybrid", True)
        if hybrid:
            return self.hybrid_search(
                query=query,
                top_k=top_k,
                strategy=params.get("strategy", "rrf"),
                rrf_k=params.get("rrf_k", 60),
                alpha=params.get("alpha", 0.5),
            )
        return self.search(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        段落感知的分层切分策略：

        1. 先按空行切分为段落
        2. 段落长度 ≤ chunk_size → 完整保留为一个 chunk
        3. 段落长度 > chunk_size → 在该段落内按句子切分（递归降级）
        4. 相邻短段落合并（合并后总长不超过 chunk_size 时合并，减少碎片）

        Args:
            text:     原始文本
            chunk_size: 每片最大字符数（默认 500）
            overlap:    相邻 chunk 的重叠字符数（默认 50）

        Returns:
            分片后的文本列表
        """
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        import re

        # ── 1. 按空行切分为段落 ──
        paragraphs = re.split(r'\n{2,}', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        if not paragraphs:
            return []

        # ── 2. 处理每个段落 ──
        processed = []
        for para in paragraphs:
            if len(para) <= chunk_size:
                # 段落没超限 → 完整保留
                processed.append(para)
            else:
                # 超长段落 → 段落内按句子切分
                para_chunks = self._chunk_by_sentences(para, chunk_size, overlap)
                processed.extend(para_chunks)

        # ── 3. 合并相邻短段落（合并后不超 chunk_size 就合并） ──
        if len(processed) <= 1:
            return processed

        merged = [processed[0]]
        for chunk in processed[1:]:
            if len(merged[-1]) + 1 + len(chunk) <= chunk_size:
                merged[-1] = f"{merged[-1]}\n\n{chunk}"
            else:
                merged.append(chunk)

        return merged

    @staticmethod
    def _chunk_by_sentences(text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        超长段落内的句子级切分（保持原有的句子完整性逻辑）。
        仅在被 _chunk_text 判定段落超限时调用。
        """
        import re
        sentences = re.split(r'[。！？.!?\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text]

        chunks = []
        current = ""
        overlap_pool = []  # 最近几句，用于为新 chunk 构建 overlap 前缀

        for sent in sentences:
            candidate = f"{current}。{sent}" if current else sent

            if len(candidate) > chunk_size:
                if current:
                    chunks.append(current)

                # 从 overlap_pool 尾部提取不超过 overlap 字符的尾句
                prefix = ""
                chars = 0
                for s in reversed(overlap_pool):
                    cost = len(s) + 1  # 句子本身 + 句号
                    if chars + cost > overlap:
                        break
                    prefix = f"{s}。{prefix}" if prefix else s
                    chars += cost

                current = f"{prefix}。{sent}" if prefix else sent
            else:
                current = candidate

            overlap_pool.append(sent)
            total = sum(len(s) for s in overlap_pool)
            while total > chunk_size and len(overlap_pool) > 1:
                total -= len(overlap_pool.pop(0))

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（客服场景常见问题）。"""
        default_docs = [
            {
                "title": "退款政策",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "content": (
                    "常见技术问题排查。"
                    "应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
