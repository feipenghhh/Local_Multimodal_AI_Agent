from chromadb import PersistentClient

DB_DIR = "data/index"
COLLECTION_NAME = "papers"

client = PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

# 获取所有文档的元数据
docs = collection.get(include=["metadatas"])

# 使用 set 去重
file_names = set()
for meta in docs["metadatas"]:
    if "file" in meta:
        file_names.add(meta["file"])

# 打印唯一文件名
print("=== 已索引论文文件名 ===")
for f in sorted(file_names):
    print("-", f)
