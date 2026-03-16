"""
=============================================================
  Hierarchical Chunking (Structure-aware Chunking) in RAG
  Demo với dữ liệu ngắn - không cần GPU / API key
=============================================================
"""

import re
import math
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# 1. DỮ LIỆU MẪU (giả lập 1 chương sách Python)
# ─────────────────────────────────────────────────────────────

SAMPLE_DOCUMENT = """
# Chương 3: Hàm trong Python

Hàm là một khối lệnh được đặt tên, có thể tái sử dụng nhiều lần trong chương trình.
Việc sử dụng hàm giúp code gọn gàng, dễ bảo trì và tránh lặp lại logic.

## 3.1 Định nghĩa hàm cơ bản

Để định nghĩa hàm trong Python, ta dùng từ khóa `def` theo sau là tên hàm và dấu ngoặc đơn.
Thân hàm được thụt lề vào bên trong.
Hàm có thể trả về giá trị bằng lệnh `return`.
Nếu không có `return`, hàm mặc định trả về `None`.

## 3.2 Tham số và đối số

Tham số là biến được khai báo trong định nghĩa hàm.
Đối số là giá trị thực tế truyền vào khi gọi hàm.
Python hỗ trợ tham số mặc định, cho phép bỏ qua đối số khi gọi hàm.
Ví dụ: `def chao(ten="bạn"): print(f"Xin chào {ten}")`.
Ngoài ra, `*args` cho phép truyền số lượng đối số vị trí không xác định.
`**kwargs` cho phép truyền số lượng đối số từ khóa không xác định.

## 3.3 Hàm lambda

Hàm lambda là hàm ẩn danh, được định nghĩa trên một dòng duy nhất.
Cú pháp: `lambda tham_số: biểu_thức`.
Lambda thường dùng kết hợp với `map()`, `filter()`, `sorted()`.
Ví dụ: `sorted(ds, key=lambda x: x["tuoi"])` sắp xếp danh sách theo tuổi.
""".strip()


# ─────────────────────────────────────────────────────────────
# 2. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id: str
    level: int          # 1=chapter, 2=section, 3=paragraph (leaf)
    title: str
    content: str
    parent_id: Optional[str] = None
    children_ids: list = field(default_factory=list)
    embedding: Optional[list] = None

    def token_count(self):
        return len(self.content.split())

    def __repr__(self):
        return (f"Chunk(level={self.level}, title='{self.title}', "
                f"tokens={self.token_count()}, children={len(self.children_ids)})")


# ─────────────────────────────────────────────────────────────
# 3. PARSER: Tách tài liệu theo cấu trúc Markdown
# ─────────────────────────────────────────────────────────────

class HierarchicalParser:
    """Parse Markdown document thành cây phân cấp chunks."""

    def parse(self, text: str) -> dict[str, Chunk]:
        chunks: dict[str, Chunk] = {}
        lines = text.split("\n")

        current_chapter: Optional[Chunk] = None
        current_section: Optional[Chunk] = None
        current_paragraphs: list[str] = []

        def flush_paragraphs(parent: Optional[Chunk]):
            """Gộp các câu đang chờ thành leaf chunks."""
            if not parent or not current_paragraphs:
                return
            # Tách thành từng câu (leaf)
            sentences = []
            for para in current_paragraphs:
                for s in re.split(r'(?<=[.!?])\s+', para.strip()):
                    s = s.strip()
                    if len(s) > 10:
                        sentences.append(s)

            # Nhóm 2 câu liền nhau thành 1 leaf chunk (sliding window nhỏ)
            i = 0
            idx = len([c for c in chunks.values() if c.level == 3])
            while i < len(sentences):
                group = sentences[i:i+2]
                leaf_content = " ".join(group)
                leaf_id = f"leaf_{parent.id}_{idx}"
                leaf = Chunk(
                    id=leaf_id,
                    level=3,
                    title=f"[Leaf] {parent.title} (đoạn {idx+1})",
                    content=leaf_content,
                    parent_id=parent.id,
                )
                chunks[leaf_id] = leaf
                parent.children_ids.append(leaf_id)
                idx += 1
                i += 2

        for line in lines:
            line = line.rstrip()
            if not line:
                continue

            if line.startswith("# "):  # Level 1
                flush_paragraphs(current_section or current_chapter)
                current_paragraphs = []
                title = line[2:].strip()
                chunk_id = self._make_id(title)
                current_chapter = Chunk(id=chunk_id, level=1, title=title, content=title)
                chunks[chunk_id] = current_chapter
                current_section = None

            elif line.startswith("## "):  # Level 2
                flush_paragraphs(current_section or current_chapter)
                current_paragraphs = []
                title = line[3:].strip()
                chunk_id = self._make_id(title)
                current_section = Chunk(
                    id=chunk_id, level=2, title=title, content=title,
                    parent_id=current_chapter.id if current_chapter else None,
                )
                chunks[chunk_id] = current_section
                if current_chapter:
                    current_chapter.children_ids.append(chunk_id)

            else:  # Nội dung thường → tích lũy
                current_paragraphs.append(line)

        flush_paragraphs(current_section or current_chapter)

        # Tổng hợp content cho level 1 & 2 từ leaf children
        for chunk in chunks.values():
            if chunk.level in (1, 2) and chunk.children_ids:
                leaf_texts = []
                for cid in chunk.children_ids:
                    child = chunks.get(cid)
                    if child:
                        if child.level == 3:
                            leaf_texts.append(child.content)
                        else:
                            # child là level 2 → lấy leaf của nó
                            for gcid in child.children_ids:
                                gc = chunks.get(gcid)
                                if gc:
                                    leaf_texts.append(gc.content)
                chunk.content = chunk.title + "\n\n" + "\n".join(leaf_texts)

        return chunks

    def _make_id(self, text: str) -> str:
        slug = re.sub(r'[^a-zA-Z0-9_àáâãèéêìíòóôõùúýăđơưạặ]', '_', text.lower())
        short = slug[:30].strip('_')
        h = hashlib.md5(text.encode()).hexdigest()[:4]
        return f"{short}_{h}"


# ─────────────────────────────────────────────────────────────
# 4. FAKE EMBEDDER (TF-IDF đơn giản, không cần model)
# ─────────────────────────────────────────────────────────────

class SimpleEmbedder:
    """
    TF-IDF bag-of-words embedding đơn giản.
    Trong production thay bằng: OpenAI, sentence-transformers, v.v.
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]):
        # Đếm document frequency
        df: dict[str, int] = defaultdict(int)
        tokenized = [self._tokenize(t) for t in texts]
        for tokens in tokenized:
            for w in set(tokens):
                df[w] += 1
        N = len(texts)
        # Build vocab & IDF
        for w, count in df.items():
            self.vocab[w] = len(self.vocab)
            self.idf[w] = math.log((N + 1) / (count + 1)) + 1  # smooth

    def embed(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        vec = [0.0] * len(self.vocab)
        tf: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for w, count in tf.items():
            if w in self.vocab:
                idx = self.vocab[w]
                vec[idx] = (count / len(tokens)) * self.idf.get(w, 1.0)
        # L2 normalize
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def _tokenize(self, text: str) -> list[str]:
        # Tách từ đơn giản (unicode-friendly)
        return re.findall(r'\b\w+\b', text.lower())

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x*y for x, y in zip(a, b))
        return dot  # đã normalize → dot = cosine


# ─────────────────────────────────────────────────────────────
# 5. HIERARCHICAL RETRIEVER
# ─────────────────────────────────────────────────────────────

class HierarchicalRetriever:
    """
    Index leaf chunks → tìm kiếm → trả về parent chunk.
    Đây là trái tim của kỹ thuật Hierarchical Chunking.
    """

    def __init__(self, chunks: dict[str, Chunk], embedder: SimpleEmbedder):
        self.chunks = chunks
        self.embedder = embedder
        # Chỉ index leaf chunks (level 3)
        self.leaf_ids = [cid for cid, c in chunks.items() if c.level == 3]

    def search(self, query: str, top_k: int = 2, return_level: int = 2) -> list[dict]:
        """
        1. Embed query
        2. Tìm top_k leaf chunks giống nhất
        3. Trả về parent tương ứng (level = return_level)

        return_level=2 → trả section (mục)
        return_level=1 → trả chapter (chương) - ngữ cảnh rộng hơn
        """
        q_vec = self.embedder.embed(query)

        # Tính similarity với tất cả leaf
        scored = []
        for lid in self.leaf_ids:
            leaf = self.chunks[lid]
            if leaf.embedding is None:
                continue
            sim = self.embedder.cosine_similarity(q_vec, leaf.embedding)
            scored.append((sim, lid))

        scored.sort(reverse=True)
        top_leaves = scored[:top_k]

        results = []
        seen_parents = set()

        for sim, lid in top_leaves:
            leaf = self.chunks[lid]

            # Leo lên cây để tìm parent đúng cấp
            parent = self._find_ancestor(leaf, target_level=return_level)

            result = {
                "query": query,
                "matched_leaf": {
                    "id": lid,
                    "content": leaf.content,
                    "similarity": round(sim, 4),
                },
                "returned_context": {
                    "id": parent.id if parent else lid,
                    "level": parent.level if parent else leaf.level,
                    "title": parent.title if parent else leaf.title,
                    "content": parent.content if parent else leaf.content,
                    "token_count": parent.token_count() if parent else leaf.token_count(),
                },
                "deduped": parent.id in seen_parents if parent else False,
            }
            if parent:
                seen_parents.add(parent.id)
            results.append(result)

        return results

    def _find_ancestor(self, chunk: Chunk, target_level: int) -> Optional[Chunk]:
        """Đi từ leaf lên parent cho đến khi đạt level mong muốn."""
        current = chunk
        while current.level > target_level and current.parent_id:
            parent = self.chunks.get(current.parent_id)
            if parent is None:
                break
            current = parent
        return current if current.level == target_level else chunk


# ─────────────────────────────────────────────────────────────
# 6. MAIN DEMO
# ─────────────────────────────────────────────────────────────

def print_tree(chunks: dict[str, Chunk]):
    """In cây phân cấp đẹp."""
    print("\n" + "═"*60)
    print("  CÂY PHÂN CẤP CHUNKS")
    print("═"*60)
    indent = {1: "", 2: "  ├─ ", 3: "  │    └─ "}
    for chunk in sorted(chunks.values(), key=lambda c: (c.level, c.title)):
        prefix = indent.get(chunk.level, "")
        label = ["", "📕 L1", "📄 L2", "🔹 L3"][chunk.level]
        print(f"{prefix}{label} [{chunk.id[:12]}] \"{chunk.title[:45]}\"  "
              f"({chunk.token_count()} tokens)")
    print()


def print_search_result(result: dict, idx: int):
    print(f"\n{'─'*60}")
    print(f"  KẾT QUẢ #{idx+1}: \"{result['query']}\"")
    print(f"{'─'*60}")

    leaf = result["matched_leaf"]
    ctx  = result["returned_context"]

    print(f"\n  🔍 Leaf chunk khớp nhất  (similarity={leaf['similarity']}):")
    print(f"     \"{leaf['content'][:120]}...\"" if len(leaf['content'])>120 else f"     \"{leaf['content']}\"")

    print(f"\n  📄 Trả về cho LLM  →  [{ctx['title'][:50]}]  ({ctx['token_count']} tokens)")
    content_preview = ctx['content'].replace('\n', ' ')[:200]
    print(f"     {content_preview}...")

    ratio = ctx["token_count"] / max(len(leaf["content"].split()), 1)
    print(f"\n  ℹ️  Context mở rộng {ratio:.1f}x so với leaf chunk đơn lẻ")


def main():
    print("\n" + "█"*60)
    print("  HIERARCHICAL CHUNKING — DEMO ĐẦY ĐỦ")
    print("█"*60)

    # ── BƯỚC 1: Parse ─────────────────────────────────────────
    print("\n[BƯỚC 1] Parse tài liệu → cây phân cấp")
    parser = HierarchicalParser()
    chunks = parser.parse(SAMPLE_DOCUMENT)
    print(f"  → Tổng số chunks: {len(chunks)}")
    print(f"  → Level 1 (chapter) : {sum(1 for c in chunks.values() if c.level==1)}")
    print(f"  → Level 2 (section) : {sum(1 for c in chunks.values() if c.level==2)}")
    print(f"  → Level 3 (leaf)    : {sum(1 for c in chunks.values() if c.level==3)}")
    print_tree(chunks)

    # ── BƯỚC 2: Embed ─────────────────────────────────────────
    print("[BƯỚC 2] Tạo embeddings cho leaf chunks")
    leaf_chunks = [c for c in chunks.values() if c.level == 3]

    embedder = SimpleEmbedder()
    all_texts = [c.content for c in chunks.values()]
    embedder.fit(all_texts)

    for leaf in leaf_chunks:
        leaf.embedding = embedder.embed(leaf.content)

    print(f"  → Đã embed {len(leaf_chunks)} leaf chunks vào vector DB")
    print(f"  → Kích thước vector: {len(leaf_chunks[0].embedding)} dims (vocab size)")

    # ── BƯỚC 3: Retrieve ──────────────────────────────────────
    print("\n[BƯỚC 3] Retrieval — search leaf, retrieve parent")
    retriever = HierarchicalRetriever(chunks, embedder)

    queries = [
        "tham số mặc định khi gọi hàm",
        "cách dùng lambda với sorted",
        "hàm trả về giá trị như thế nào",
    ]

    for i, query in enumerate(queries):
        results = retriever.search(query, top_k=1, return_level=2)
        print_search_result(results[0], i)

    # ── BƯỚC 4: So sánh Flat vs Hierarchical ──────────────────
    print("\n\n" + "═"*60)
    print("  SO SÁNH: FLAT CHUNKING vs HIERARCHICAL CHUNKING")
    print("═"*60)

    query_compare = "tham số mặc định khi gọi hàm"
    results = retriever.search(query_compare, top_k=1, return_level=2)
    ctx = results[0]["returned_context"]
    leaf_txt = results[0]["matched_leaf"]["content"]

    print(f"\n  Query: \"{query_compare}\"")

    print(f"\n  ❌ Flat chunking  gửi LLM ({len(leaf_txt.split())} tokens):")
    print(f"     \"{leaf_txt}\"")

    print(f"\n  ✅ Hierarchical   gửi LLM ({ctx['token_count']} tokens):")
    for line in ctx["content"].split("\n")[:8]:
        if line.strip():
            print(f"     {line}")

    print(f"\n  📊 Hierarchical cung cấp {ctx['token_count']/max(len(leaf_txt.split()),1):.1f}x")
    print(f"     ngữ cảnh hơn → LLM hiểu mối quan hệ giữa các loại tham số\n")


if __name__ == "__main__":
    main()