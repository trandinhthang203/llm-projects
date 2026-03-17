import os
from dataclasses import dataclass, field
from typing import List

# ── ENV ──────────────────────────────────────────────────────────────────────
# Đặt OPENAI_API_KEY trong .env hoặc biến môi trường nếu dùng OpenAI
# Để demo không cần key, ta dùng HuggingFace embedding miễn phí

@dataclass
class EmbeddingConfig:
    # Model đa ngôn ngữ, rất tốt cho tiếng Việt
    model_name: str = "BAAI/bge-m3"
    # Nếu muốn nhanh hơn, dùng "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cpu"            # "cuda" nếu có GPU
    normalize_embeddings: bool = True

@dataclass
class ChunkConfig:
    # ── Tầng 1: Metadata summary (nhỏ, dùng cho discovery) ──────────────────
    metadata_max_tokens: int = 300

    # ── Tầng 2: Section-level chunks (trung bình, dùng cho retrieval) ────────
    section_max_tokens: int = 600
    section_overlap_tokens: int = 0     # 0 vì sections có ranh giới rõ ràng

    # ── Tầng 3: Full document (lớn, dùng làm parent context) ────────────────
    full_doc_max_tokens: int = 2000

@dataclass
class VectorDBConfig:
    persist_dir: str = "./chroma_db"
    collection_metadata_name: str = "tthc_metadata"    # Tầng 1
    collection_section_name: str  = "tthc_sections"    # Tầng 2
    collection_full_name: str     = "tthc_full"        # Tầng 3
    distance_metric: str = "cosine"

@dataclass
class RetrievalConfig:
    # Số chunks trả về ở mỗi tầng
    top_k_sections: int = 5
    top_k_metadata: int = 3
    # Ngưỡng similarity score (0-1)
    min_similarity_score: float = 0.35
    # Có fetch parent doc không khi score thấp
    fallback_to_parent: bool = True
    parent_fallback_threshold: float = 0.45

@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # LLM config (OpenAI hoặc local)
    llm_provider: str = "openai"        # "openai" | "local"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    temperature: float = 0.1

# ── Section anchors ───────────────────────────────────────────────────────────
# Mapping section_key → các pattern heading có thể gặp trong raw text
SECTION_PATTERNS = {
    "trinh_tu":          [r"##\s*Trình tự thực hiện", r"Trình tự thực hiện\s*\n"],
    "thanh_phan_ho_so":  [r"##\s*Thành phần hồ sơ", r"Thành phần hồ sơ\s*[:,\n]"],
    "can_cu_phap_ly":    [r"##\s*Căn cứ pháp lý", r"Căn cứ pháp lý\s*\n"],
    "doi_tuong":         [r"##\s*Đối tượng thực hiện", r"Đối tượng thực hiện\s*"],
    "dieu_kien":         [r"##\s*Điều kiện thực hiện", r"Điều kiện thực hiện\s*"],
    "ket_qua":           [r"##\s*Kết quả thực hiện", r"Kết quả thực hiện\s*"],
    "dia_chi":           [r"Địa chỉ tiếp nhận hồ sơ", r"##\s*Địa chỉ"],
    "phi_le_phi":        [r"Lệ phí\s*", r"##\s*Phí\s*"],
    "co_quan":           [r"Cơ quan thực hiện", r"##\s*Cơ quan"],
}

SECTION_DISPLAY_NAMES = {
    "trinh_tu":         "Trình tự thực hiện",
    "thanh_phan_ho_so": "Thành phần hồ sơ",
    "can_cu_phap_ly":   "Căn cứ pháp lý",
    "doi_tuong":        "Đối tượng thực hiện",
    "dieu_kien":        "Điều kiện thực hiện",
    "ket_qua":          "Kết quả thực hiện",
    "dia_chi":          "Địa chỉ tiếp nhận hồ sơ",
    "phi_le_phi":       "Phí và Lệ phí",
    "co_quan":          "Cơ quan thực hiện",
}

# Normalise domain names (tên lĩnh vực đa dạng trong PDF → key chuẩn)
DOMAIN_NORMALIZATION = {
    "xây dựng nhà ở": "Xây dựng",
    "hoạt động xây dựng": "Xây dựng",
    "việc làm": "Lao động - Việc làm",
    "lao động": "Lao động - Việc làm",
    "đất đai": "Đất đai",
    "lý lịch tư pháp": "Tư pháp",
    "hộ tịch": "Hộ tịch",
    "du lịch": "Du lịch",
    "thủy sản": "Thủy sản",
    "y tế": "Y tế - Dược phẩm",
    "dược": "Y tế - Dược phẩm",
    "an toàn thực phẩm": "An toàn thực phẩm",
    "môi trường": "Tài nguyên - Môi trường",
    "tài nguyên": "Tài nguyên - Môi trường",
    "giao thông": "Giao thông vận tải",
    "đường bộ": "Giao thông vận tải",
    "kinh doanh": "Kinh doanh - Đầu tư",
    "đầu tư": "Kinh doanh - Đầu tư",
    "bất động sản": "Kinh doanh - Đầu tư",
    "bảo trợ xã hội": "Xã hội",
    "người có công": "Xã hội",
    "giáo dục": "Giáo dục",
    "văn hóa": "Văn hóa - Thể thao",
    "chứng thực": "Tư pháp",
    "ngoại vụ": "Ngoại vụ",
    "khoa học": "Khoa học - Công nghệ",
    "thương mại": "Công Thương",
}