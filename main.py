import argparse
import os
import shutil
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from tqdm import tqdm
import subprocess
import warnings
from pypdf.errors import PdfReadWarning
import torch
import open_clip
from PIL import Image

import os

# 将模型缓存重定向到 /data 分区
os.environ['HF_HOME'] = '/data/pengfei/.cache/huggingface'
os.environ['XDG_CACHE_HOME'] = '/data/pengfei/.cache'

# ======================
# 配置
# ======================
PAPER_DIR = "data/papers"
RAW_DIR = "data/papers_raw"
DB_DIR = "data/index"
COLLECTION_NAME = "papers"
IMAGE_DIR = "data/images"
IMAGE_COLLECTION_NAME = "images"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
LLM_MODEL = "qwen2:1.5b"

EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

client = PersistentClient(path=DB_DIR)

# 加载 CLIP 模型
# CLIP 使得文字和图片可以直接对比
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

def get_image_collection():
    return client.get_or_create_collection(IMAGE_COLLECTION_NAME)

# ======================
# 索引图片
# ======================
def index_images(folder_path):
    """扫描文件夹并对图片进行向量化"""
    col = get_image_collection()
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(image_files, desc="Indexing Images"):
        path = os.path.join(folder_path, filename)
        try:
            image = preprocess(Image.open(path)).unsqueeze(0)
            with torch.no_grad():
                # 提取图片向量
                image_features = clip_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                emb = image_features.tolist()[0]
            
            col.add(
                embeddings=[emb],
                metadatas=[{"path": path, "filename": filename}],
                ids=[filename]
            )
        except Exception as e:
            print(f"Error indexing {filename}: {e}")

# ======================
# 以文搜图
# ======================
def get_image_collection():
    # 显式指定使用余弦相似度，避免 L2 导致距离超过 1
    return client.get_or_create_collection(
        IMAGE_COLLECTION_NAME, 
        metadata={"hnsw:space": "cosine"}
    )


def search_image_by_text(query, top_k=1):
    col = get_image_collection()
    
    # 文本向量化并归一化
    text = clip_tokenizer([query])
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        q_emb = text_features.cpu().numpy().tolist()[0]
    
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["metadatas"]
    )

    if not res["metadatas"] or not res["metadatas"][0]:
        return
    print(f"\n>>> 🖼️ 搜索描述 '{query}' 的匹配结果:")
    print("-" * 60)
    for meta in res["metadatas"][0]:
        print(meta['filename'])
    print("-" * 60)
# ======================
# 工具
# ======================
def get_collection():
    return client.get_or_create_collection(COLLECTION_NAME)

def load_pdf_by_page(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages

def chunk_text(text, size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks



def call_qwen(prompt, model="qwen2:1.5b"):
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ollama CLI Error: {result.stderr}")
        return result.stdout.strip()
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")


# ======================
# 自动分类
# ======================
def classify_paper(text):
    prompt = f"""
请判断以下论文主要研究方向，只返回一个标签：
CV / NLP / RL / Other

论文内容：
{text[:2000]}
"""
    label = call_qwen(prompt)
    return label if label in ["CV", "NLP", "RL"] else "Other"

# ======================
# 添加论文
# ======================
def add_paper(pdf_path):
    pages = load_pdf_by_page(pdf_path)
    full_text = " ".join(p["text"] for p in pages)

    category = classify_paper(full_text)
    save_dir = os.path.join(PAPER_DIR, category)
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(pdf_path)
    save_path = os.path.join(save_dir, filename)
    shutil.copy(pdf_path, save_path)

    collection = get_collection()

    docs, embs, metas, ids = [], [], [], []

    for p in tqdm(pages, desc="Indexing"):
        for i, chunk in enumerate(chunk_text(p["text"])):
            docs.append(chunk)
            embs.append(EMB_MODEL.encode(chunk).tolist())
            metas.append({"file": filename, "page": p["page"], "category": category})
            ids.append(f"{filename}_p{p['page']}_c{i}")

    collection.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)

    print(f"[DONE] {filename} → {category}, chunks={len(docs)}")

# ======================
# 语义搜索 + 返回最相关论文及页码
# ======================
def search_paper(query):
    q_emb = EMB_MODEL.encode(query).tolist()
    col = get_collection()

    # 查询 top 1 最相关论文
    res = col.query(
        query_embeddings=[q_emb],
        n_results=1,
        include=["documents", "metadatas", "distances"]
    )

    if not res["documents"][0]:
        print("未找到相关论文。")
        return

    documents = res["documents"][0]
    metadatas = res["metadatas"][0]

    # 获取最相关论文的文件名
    top_file = metadatas[0]["file"]

    # 收集属于该论文的所有页码
    pages = set()
    for meta in metadatas:
        if meta["file"] == top_file:
            pages.add(meta["page"])
    pages = sorted(pages)

    # 输出结果
    print(f"\n【最相关论文】{top_file}")
    print("【相关页码】")
    for p in pages:
        print("-", p)




# ======================
# 语义搜索 + 文件索引 
# ======================
def list_files(query, top_k_files=5, search_depth=50):
    """
    列出与查询最相关的文件列表。
    原理：检索 Top N 个相关片段，统计每个文件包含的片段数量，以此作为相关性打分。
    
    Args:
        query (str): 搜索语句
        top_k_files (int): 返回的文件数量
        search_depth (int): 检索的片段总池大小（越大越精准，但略慢）
    """
    q_emb = EMB_MODEL.encode(query).tolist()
    col = get_collection()

    # 1. 扩大搜索范围，获取更多相关片段以统计分布
    res = col.query(
        query_embeddings=[q_emb],
        n_results=search_depth, 
        include=["metadatas", "distances"]
    )

    if not res["metadatas"] or not res["metadatas"][0]:
        print(f"未找到与 '{query}' 相关的论文。")
        return []

    # 2. 统计文件出现频率 (Hit Count)
    file_stats = {}  # {filename: {'count': int, 'category': str}}
    
    for meta in res["metadatas"][0]:
        fname = meta["file"]
        cat = meta.get("category", "Unknown")
        
        if fname not in file_stats:
            file_stats[fname] = {'count': 0, 'category': cat}
        
        file_stats[fname]['count'] += 1

    # 3. 排序：按命中次数倒序 (命中次数越多，说明文中相关内容越多)
    ranked_files = sorted(file_stats.items(), key=lambda x: x[1]['count'], reverse=True)

    # 4. 格式化输出
    print(f"\n>>> 🔍 关键词 '{query}' 的搜索结果 (Top {top_k_files}):")
    print("-" * 60)
    print(f"{'Category':<10} | {'Rel. Score':<10} | {'Filename'}")
    print("-" * 60)

    result_filenames = []
    
    for fname, stats in ranked_files[:top_k_files]:
        # 简单的归一化分数展示 (基于 search_depth)
        score_display = f"{stats['count']}" 
        print(f"[{stats['category']:<8}] | {score_display:<10} | {fname}")
        result_filenames.append(fname)
        
    print("-" * 60)
    
    return result_filenames






# ======================
# 批量整理
# ======================
def organize_all(folder):
    for f in os.listdir(folder):
        if f.endswith(".pdf"):
            add_paper(os.path.join(folder, f))

# ======================
# CLI
# ======================
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    add_parser = sub.add_parser("add_paper")
    add_parser.add_argument("path")
    add_parser.add_argument(
        "--topics",
        type=str,
        default="CV,NLP,RL",
    )

    sub.add_parser("search_paper").add_argument("query")
    sub.add_parser("organize_all").add_argument("folder")
    list_parser = sub.add_parser("list_files")
    list_parser.add_argument("query", type=str, help="搜索关键词")
    list_parser.add_argument("--top_k", type=int, default=5, help="显示结果数量")
    


    # 索引图片的命令
    img_idx_parser = sub.add_parser("index_images")
    img_idx_parser.add_argument("folder", help="图片文件夹路径")
    
    # 搜索图片的命令
    img_search_parser = sub.add_parser("search_image")
    img_search_parser.add_argument("query", help="搜索关键词")
    img_search_parser.add_argument("--top_k", type=int, default=1, help="输出结果数量")

    args = parser.parse_args()
    if args.cmd == "add_paper":
        add_paper(args.path)
    elif args.cmd == "search_paper":
        search_paper(args.query)
    elif args.cmd == "list_files":
        list_files(args.query, top_k_files=args.top_k)
    elif args.cmd == "organize_all":
        organize_all(args.folder)
    elif args.cmd == "index_images":
        index_images(args.folder)
    elif args.cmd == "search_image":
        search_image_by_text(args.query, top_k=args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
