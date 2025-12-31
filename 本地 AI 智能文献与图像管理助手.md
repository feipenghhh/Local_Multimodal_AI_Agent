

# 📚🖼️ 本地 AI 智能文献与图像管理助手

**Local Multimodal AI Agent**

## 1. 项目简介（Project Introduction）

本项目是一个基于 **Python 的本地多模态 AI 智能文献与图像管理助手**，旨在解决本地大量论文 PDF 与图像素材难以检索、难以整理的问题。

不同于传统基于**文件名或关键词匹配**的搜索方式，本项目利用 **自然语言处理（NLP）+ 多模态表示学习（CLIP）+ 向量数据库（ChromaDB）**，实现：

- 📄 **论文内容级语义搜索**
- 🗂️ **论文自动分类与批量整理**
- 🖼️ **以文搜图（Text-to-Image Retrieval）**

项目支持 **完全本地化部署**，也支持通过 **Ollama 调用本地大语言模型（Qwen2）**，在不依赖云端 API 的情况下完成智能理解与分类，适合课程作业、科研学习与个人知识库管理。

## 2. 核心功能（Core Features）

### 2.1 📄 智能文献管理

#### （1）语义搜索（Semantic Search）

- 支持使用自然语言查询论文内容
  例如：

  ```
  Transformer 的核心架构是什么？
  ```

- 系统基于论文 **正文语义向量** 返回最相关论文

- 支持返回：

  - 📄 最相关论文文件名
  - 📑 对应的页码（Page-level 定位）

#### （2）自动分类与整理

- **单文件处理**
  - 添加新论文时，自动分析内容
  - 使用本地 LLM（Qwen2）判断研究方向：
    - `CV / NLP / RL / Other`
  - 自动移动至对应子文件夹
- **批量整理**
  - 对已有杂乱 PDF 文件夹进行“一键整理”
  - 自动扫描 → 分类 → 建立索引

#### （3）文件索引模式（File-level Retrieval）

- 仅返回与查询最相关的论文文件列表
- 适合快速定位需要阅读的文献集合

### 2.2 🖼️ 智能图像管理

#### 以文搜图（Text-to-Image Retrieval）

- 利用 **CLIP 多模态模型**

- 支持通过自然语言描述搜索本地图片库，例如：

  ```
  海边的日落
  ```

- 返回最匹配的图片文件名

- 支持 Top-K 结果输出

## 3. 技术选型与模型说明（Technical Stack）

### 3.1 文献处理与语义理解

| 模块       | 技术                                      |
| ---------- | ----------------------------------------- |
| PDF 解析   | `pypdf`                                   |
| 文本嵌入   | `SentenceTransformers (all-MiniLM-L6-v2)` |
| 向量数据库 | `ChromaDB (PersistentClient)`             |
| 本地 LLM   | `Qwen2-1.5B (via Ollama)`                 |

- 文本按 **页 → 分块（chunk）** 建立向量索引
- 支持页级精确定位

### 3.2 图像与多模态检索

| 模块          | 技术                  |
| ------------- | --------------------- |
| 图像-文本对齐 | `OpenCLIP (ViT-B-32)` |
| 相似度度量    | Cosine Similarity     |
| 图像数据库    | ChromaDB              |

### 3.3 系统特点

- ✅ 完全本地运行（无需云 API）
- ✅ 模块化设计，模型可替换
- ✅ 支持 CPU / GPU
- ✅ 向量数据库持久化存储

## 4. 项目结构（Project Structure）

```
.
├── main.py                  # 项目统一入口
├── data/
│   ├── papers_raw/           # 原始 PDF（待整理）
│   ├── papers/               # 按类别整理后的论文
│   │   ├── CV/
│   │   ├── NLP/
│   │   ├── RL/
│   │   └── Other/
│   ├── images/               # 本地图像库
│   └── index/                # ChromaDB 向量索引
└── README.md
```

## 5. 环境配置（Environment）

### 5.1 基本环境

- 操作系统：Windows / macOS / Linux
- Python：**3.8 及以上**
- 内存：建议 **8GB+**

### 5.2 依赖安装

```
pip install pypdf sentence-transformers chromadb open-clip-torch pillow tqdm
```

### 5.3 本地 LLM（可选）

本项目使用 **Ollama** 调用 Qwen2：

```
ollama pull qwen2:1.5b
```

## 6. 使用说明（Usage）

⚠️ **所有功能统一通过 main.py 调用**

### 6.1 添加并自动分类论文

```
python main.py add_paper path/to/paper.pdf --topics "CV,NLP,RL"
```

功能：

- 自动分类（CV / NLP / RL / Other）
- 自动移动文件
- 建立向量索引

### 6.2 批量整理论文文件夹

```
python main.py organize_all data/papers_raw
```

### 6.3 语义搜索论文（返回页码）

```
python main.py search_paper "self-attention mechanism"
```

输出示例：

```
【最相关论文】Attention_is_All_You_Need.pdf
【相关页码】
- 2
- 3
- 5
```

### 6.4 文件级索引搜索

```
python main.py list_files "domain adaptation" --top_k 5
```

### 6.5 索引本地图像

```
python main.py index_images data/images
```

### 6.6 以文搜图

```
python main.py search_image "海边的日落" --top_k 3
```

## 7. 演示截图

### 7.1添加并自动分类论文

```
python main.py add_paper "/data/pengfei/Local_Multimodal_AI_Agent/论文/CV/G-NAS Generalizable Neural Architecture Search for Single Domain Generalization Object Detection.pdf" --topics "CV,NLP,RL"
```

![img](https://cdn.nlark.com/yuque/0/2025/png/40646111/1767151596202-ea3c3533-e8e3-4ad0-b33c-121843e30b6b.png) 

添加后文件夹：

![img](https://cdn.nlark.com/yuque/0/2025/png/40646111/1767151614102-8c1f9e9e-a106-4aec-bbd7-2400e88309c2.png) 

### 7.2批量整理论文文件夹

```
python main.py organize_all data/papers_raw
```

整理前文件夹：

![img](https://cdn.nlark.com/yuque/0/2025/png/40646111/1767189474200-d87a152d-8de3-47cf-978d-447c958c140c.png) 

整理后：![img](https://cdn.nlark.com/yuque/0/2025/png/40646111/1767189595487-aef7d3b1-49a4-4119-ad0d-a412a643788a.png) 

![img](https://cdn.nlark.com/yuque/0/2025/png/40646111/1767189617343-994c3ed3-64d9-45ec-9fde-767e8fa93892.png) 

### 7.3 语义搜索（返回页码）

```
python main.py search_paper "BERT 模型的核心架构是什么？"
```

![img](https://cdn.nlark.com/yuque/0/2025/png/40646111/1767153756052-b005eadf-bd2b-4fcb-a473-aeec931f46b0.png) 

### 7.4 文件索引搜索

```
python main.py list_files "transformer language model pre-training" --top_k 3
```

![img](https://cdn.nlark.com/yuque/0/2025/png/40646111/1767189778930-d8d697f6-68da-4e4e-8f21-f93077dc82f9.png) 

### 7.5 以文搜图

```
python main.py search_image "a photo of a cat" --top_k 1
```

![img](https://cdn.nlark.com/yuque/0/2025/png/40646111/1767191506890-9b7c4d49-7ab0-4293-9a8a-52d45f1ed393.png) 

![img](https://cdn.nlark.com/yuque/0/2025/png/40646111/1767192057800-1883c36e-9b6f-4cdb-b07e-bd96ede56744.png) 

### 7.6 删除文件

```
python delete_paper.py "Enhancing Source-Free Domain Adaptive Object Detection with Low-Confidence Pseudo Label Distillation.pdf"
```

![img](https://cdn.nlark.com/yuque/0/2025/png/40646111/1767189822478-18b0dd3c-95b5-43ee-b4f5-d8767ac61099.png) 

### 7.7 查看文件 

```
python check_index.py
```

![img](https://cdn.nlark.com/yuque/0/2025/png/40646111/1767189800032-16be36d3-beee-440c-a79a-98878c736ead.png) 