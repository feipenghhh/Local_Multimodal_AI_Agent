import os
import shutil
import argparse
from chromadb import PersistentClient

# ======================
# 配置
# ======================
PAPER_DIR = "data/papers"
DB_DIR = "data/index"
COLLECTION_NAME = "papers"

# ======================
# ChromaDB
# ======================
client = PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

# ======================
# 删除论文函数
# ======================
def delete_paper(filename):
    # 1. 删除索引
    print(f"[INFO] Deleting index for: {filename}")
    collection.delete(where={"file": filename})
    
    # 2. 删除硬盘文件
    found = False
    for root, dirs, files in os.walk(PAPER_DIR):
        if filename in files:
            file_path = os.path.join(root, filename)
            os.remove(file_path)
            print(f"[INFO] Deleted file: {file_path}")
            found = True
            break
    if not found:
        print(f"[WARN] File not found in {PAPER_DIR}")

# ======================
# CLI
# ======================
def main():
    parser = argparse.ArgumentParser(description="Delete paper and its index")
    parser.add_argument("filename", type=str, help="PDF file name to delete")
    args = parser.parse_args()
    
    delete_paper(args.filename)

if __name__ == "__main__":
    main()
