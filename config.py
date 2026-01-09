import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from qdrant_client import QdrantClient, models

load_dotenv()

# Env variables
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Embedder
embedder = NVIDIAEmbeddings(model=EMBEDDING_MODEL, truncate="END")
EMBED_DIM = len(embedder.embed_query("test"))

# Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# === Collection creation (if it does not exist) ===
COLLECTION_NAME = "paperpal_collection"

existing_collections = [c.name for c in qdrant_client.get_collections().collections]

if COLLECTION_NAME not in existing_collections:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)
    )

BASE_USER_DATA_DIR = Path("user_data")
