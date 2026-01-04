"""
RAG Demo - Streamlit Web Application

A minimal RAG system with document upload, FAISS indexing, and answer generation.
"""

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from document_process import load_and_chunk, DocumentChunk
from faiss_demo import (
    create_index,
    load_index,
    save_index,
    load_meta,
    save_meta,
    build_or_update_index,
    search,
    get_index_stats,
    clear_index,
    EMBEDDING_DIM,
    DEFAULT_TOP_K,
)
from generation import (
    generate_answer_with_sources,
    GenerationError,
)
from embedding import EmbeddingServiceError

# ========= Configuration =========
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_DIR = BASE_DIR / "faiss_out"
INDEX_PATH = INDEX_DIR / "uploads.index.faiss"
META_PATH = INDEX_DIR / "uploads.meta.json"

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx"]

# ========= Page Configuration =========
st.set_page_config(
    page_title="RAG 文档问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========= Custom CSS =========
st.markdown("""
<style>
    /* Import distinctive fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Noto+Sans+SC:wght@400;500;700&display=swap');
    
    /* Root variables - dark theme inspired by terminal aesthetics */
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-tertiary: #21262d;
        --accent-cyan: #58a6ff;
        --accent-green: #3fb950;
        --accent-orange: #d29922;
        --accent-red: #f85149;
        --text-primary: #c9d1d9;
        --text-secondary: #8b949e;
        --border-color: #30363d;
    }
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'JetBrains Mono', 'Noto Sans SC', monospace !important;
        color: var(--accent-cyan) !important;
    }
    
    h1 {
        font-size: 2.2rem !important;
        border-bottom: 2px solid var(--accent-cyan);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.4rem !important;
        text-align: center;
        border-bottom: 1px solid var(--accent-green);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--accent-cyan) !important;
        border-radius: 8px;
        padding: 1rem;
        background: rgba(88, 166, 255, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 500;
        border-radius: 6px;
        border: 1px solid var(--accent-cyan);
        background: transparent;
        color: var(--accent-cyan);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: var(--accent-cyan);
        color: var(--bg-primary);
    }
    
    /* Status boxes */
    .status-box {
        font-family: 'JetBrains Mono', monospace;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .status-success {
        background: rgba(63, 185, 80, 0.1);
        border-left-color: var(--accent-green);
        color: var(--accent-green);
    }
    
    .status-warning {
        background: rgba(210, 153, 34, 0.1);
        border-left-color: var(--accent-orange);
        color: var(--accent-orange);
    }
    
    .status-error {
        background: rgba(248, 81, 73, 0.1);
        border-left-color: var(--accent-red);
        color: var(--accent-red);
    }
    
    /* Answer box */
    .answer-box {
        font-family: 'Noto Sans SC', sans-serif;
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.08) 0%, rgba(63, 185, 80, 0.08) 100%);
        border: 1px solid var(--accent-cyan);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.8;
    }
    
    /* Source citation card */
    .source-card {
        font-family: 'JetBrains Mono', monospace;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    
    .source-header {
        color: var(--accent-green);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .source-score {
        color: var(--accent-orange);
        float: right;
    }
    
    .source-text {
        color: var(--text-secondary);
        font-size: 0.8rem;
        line-height: 1.5;
    }
    
    /* Progress indicator */
    .step-indicator {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 0.5rem;
    }
    
    .step-done { background: var(--accent-green); color: var(--bg-primary); }
    .step-active { background: var(--accent-cyan); color: var(--bg-primary); }
    .step-pending { background: var(--bg-tertiary); color: var(--text-secondary); }
    
    /* Stats display */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: var(--bg-secondary);
        border-radius: 6px;
        border: 1px solid var(--border-color);
    }
    
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        color: var(--accent-cyan);
        font-weight: 600;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem;
    }
    
    /* Text input */
    .stTextArea textarea, .stTextInput input {
        font-family: 'Noto Sans SC', sans-serif !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)


# ========= Helper Functions =========

def ensure_dirs():
    """Ensure required directories exist."""
    UPLOAD_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)


def get_safe_filename(filename: str) -> str:
    """Generate a safe filename."""
    # Keep only safe characters
    safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_')
    name, ext = os.path.splitext(filename)
    safe_name = ''.join(c if c in safe_chars else '_' for c in name)
    return f"{safe_name}{ext}"


def load_or_create_index():
    """Load existing index or create new one."""
    if INDEX_PATH.exists() and META_PATH.exists():
        index = load_index(str(INDEX_PATH))
        meta = load_meta(str(META_PATH))
        return index, meta
    else:
        index = create_index(dim=EMBEDDING_DIM)
        meta = {}
        return index, meta


def save_current_index(index, meta):
    """Save current index and metadata."""
    save_index(index, str(INDEX_PATH))
    save_meta(str(META_PATH), meta)


def process_uploaded_file(uploaded_file, progress_callback=None) -> list[DocumentChunk]:
    """Process a single uploaded file and return chunks."""
    # Save to temp file
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        if progress_callback:
            progress_callback(f"解析文件: {uploaded_file.name}")
        
        chunks = load_and_chunk(tmp_path)
        
        # Update source_file to original name
        for chunk in chunks:
            chunk.source_file = uploaded_file.name
        
        return chunks
    finally:
        os.unlink(tmp_path)


def render_status_box(message: str, status: str = "success"):
    """Render a styled status box."""
    st.markdown(f'<div class="status-box status-{status}">{message}</div>', unsafe_allow_html=True)


def render_answer_box(answer: str):
    """Render the answer in a styled box."""
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)


def render_source_card(idx: int, source: dict):
    """Render a source citation card."""
    page_info = f" · 第{source['page']+1}页" if source.get('page') is not None else ""
    score_pct = source['score'] * 100
    
    st.markdown(f"""
    <div class="source-card">
        <div class="source-header">
            [{idx}] {source['source_file']}{page_info}
            <span class="source-score">相似度: {score_pct:.1f}%</span>
        </div>
        <div class="source-text">{source['text'][:500]}{"..." if len(source['text']) > 500 else ""}</div>
    </div>
    """, unsafe_allow_html=True)


def render_stats(stats: dict):
    """Render index statistics."""
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-item">
            <div class="stat-value">{stats['total_vectors']}</div>
            <div class="stat-label">向量数量</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{stats['num_sources']}</div>
            <div class="stat-label">文档数量</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{stats['total_chunks']}</div>
            <div class="stat-label">文本块数</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ========= Main Application =========

def main():
    ensure_dirs()
    
    # Initialize session state
    if 'index' not in st.session_state:
        st.session_state.index, st.session_state.meta = load_or_create_index()
    
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    
    # ========= Sidebar =========
    with st.sidebar:
        st.markdown("# 📚 RAG 文档系统")
        
        st.markdown("---")
        
        # Index stats
        st.markdown("### 📊 索引状态")
        if st.session_state.index.ntotal > 0:
            stats = get_index_stats(st.session_state.index, st.session_state.meta)
            render_stats(stats)
            
            # List source files
            if stats['source_files']:
                st.markdown("**已索引文档:**")
                for f in stats['source_files']:
                    if f != "unknown":
                        st.markdown(f"- `{f}`")
        else:
            render_status_box("索引为空，请上传文档", "warning")
        
        st.markdown("---")
        
        # Clear index button
        if st.button("🗑️ 清空索引", use_container_width=True):
            clear_index(str(INDEX_PATH), str(META_PATH))
            st.session_state.index = create_index(dim=EMBEDDING_DIM)
            st.session_state.meta = {}
            st.session_state.last_result = None
            st.rerun()
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ 检索设置")
        top_k = st.slider("Top-K 结果数", min_value=1, max_value=20, value=5)
        score_threshold = st.slider("最低相似度", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.75rem; color: #8b949e; text-align: center;">
            RAG Demo v1.0<br>
            FAISS + OpenAI
        </div>
        """, unsafe_allow_html=True)
    
    # ========= Main Content =========
    st.markdown("# 📖 RAG 文档问答系统")
    
    # Create tabs
    tab_upload, tab_ask = st.tabs(["📤 上传文档", "💬 提问"])
    
    # ========= Upload Tab =========
    with tab_upload:
        st.markdown("### 上传文档进行索引")
        st.markdown("支持格式: **PDF**, **TXT**, **DOCX**")
        
        uploaded_files = st.file_uploader(
            "拖拽或点击上传文件",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        
        if uploaded_files:
            st.markdown(f"已选择 **{len(uploaded_files)}** 个文件")
            
            if st.button("🚀 开始处理", type="primary", use_container_width=True):
                all_chunks = []
                
                # Progress container
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_files = len(uploaded_files)
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        try:
                            # Update progress
                            progress = (i) / total_files
                            progress_bar.progress(progress)
                            status_text.markdown(f"**处理中:** `{uploaded_file.name}` ({i+1}/{total_files})")
                            
                            # Process file
                            chunks = process_uploaded_file(uploaded_file)
                            all_chunks.extend(chunks)
                            
                        except Exception as e:
                            st.error(f"处理 {uploaded_file.name} 失败: {e}")
                    
                    if all_chunks:
                        # Embedding and indexing
                        status_text.markdown("**向量化中...** 请稍候")
                        progress_bar.progress(0.8)
                        
                        try:
                            st.session_state.index, st.session_state.meta = build_or_update_index(
                                chunks=all_chunks,
                                existing_index=st.session_state.index,
                                existing_meta=st.session_state.meta,
                            )
                            
                            # Save index
                            status_text.markdown("**保存索引...**")
                            progress_bar.progress(0.95)
                            save_current_index(st.session_state.index, st.session_state.meta)
                            
                            progress_bar.progress(1.0)
                            status_text.empty()
                            
                            render_status_box(
                                f"✅ 成功处理 {len(uploaded_files)} 个文件，添加 {len(all_chunks)} 个文本块",
                                "success"
                            )
                            
                        except EmbeddingServiceError as e:
                            st.error(f"向量化服务错误: {e}")
                        except Exception as e:
                            st.error(f"索引构建失败: {e}")
                    else:
                        render_status_box("⚠️ 未能从文件中提取任何文本", "warning")
    
    # ========= Ask Tab =========
    with tab_ask:
        st.markdown("### 向文档提问")
        
        # Check if index has content
        if st.session_state.index.ntotal == 0:
            render_status_box("请先上传文档建立索引", "warning")
        else:
            # Question input
            question = st.text_area(
                "输入您的问题",
                placeholder="例如：什么是语义角色标注？",
                height=100,
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("🔍 提问", type="primary", use_container_width=True)
            
            if ask_button and question.strip():
                with st.spinner("检索中..."):
                    try:
                        # Get sidebar settings
                        threshold = score_threshold if score_threshold > 0 else None
                        
                        # Search
                        search_results = search(
                            index=st.session_state.index,
                            meta=st.session_state.meta,
                            query=question,
                            top_k=top_k,
                            score_threshold=threshold,
                        )
                        
                        if not search_results:
                            render_status_box("未找到相关内容，请调整问题或降低相似度阈值", "warning")
                        else:
                            with st.spinner("生成答案中..."):
                                result = generate_answer_with_sources(
                                    search_results=search_results,
                                    user_question=question,
                                )
                                st.session_state.last_result = result
                    
                    except EmbeddingServiceError as e:
                        st.error(f"向量化服务错误: {e}")
                    except GenerationError as e:
                        st.error(f"答案生成错误: {e}")
                    except Exception as e:
                        st.error(f"查询失败: {e}")
            
            # Display results
            if st.session_state.last_result:
                result = st.session_state.last_result
                
                st.markdown("---")
                st.markdown("### 📝 答案")
                render_answer_box(result.answer_text)
                
                st.markdown(f"*模型: {result.model}*")
                
                # Sources
                if result.used_sources:
                    st.markdown("### 📚 参考来源")
                    
                    for i, source in enumerate(result.used_sources, 1):
                        with st.expander(
                            f"[{i}] {source['source_file']} - 相似度: {source['score']*100:.1f}%",
                            expanded=False
                        ):
                            if source.get('page') is not None:
                                st.markdown(f"**页码:** {source['page']+1}")
                            st.markdown(f"**内容:**")
                            st.text(source['text'])


if __name__ == "__main__":
    main()

