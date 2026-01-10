import os
import json
import logging
from datetime import datetime
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from config import BASE_USER_DATA_DIR, COLLECTION_NAME, embedder, qdrant_client
from qdrant_client import models
from langchain_qdrant import QdrantVectorStore
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# TEXT SPLITTER
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MIN_CHUNK_LENGTH = 200
SEPARATORS = ["\n\n", "\n", ".", ";", ",", " "]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
)

# VECTORSTORE CLASS 
class PaperVectorStore:
    def __init__(self, user_id: str):
        self.user_id = user_id

        self.user_dir = BASE_USER_DATA_DIR / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)
        
        self.paper_metadata_path = self.user_dir / "paper_metadata.json"
        self.paper_metadata = self.load_paper_metadata()
        
        self.collection_name = COLLECTION_NAME
        self.qdrant_client = qdrant_client
        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=embedder,
        )

    def load_paper_metadata(self):
        """Load paper metadata from JSON file."""
        if self.paper_metadata_path.exists():
            try:
                with open(self.paper_metadata_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def save_paper_metadata(self):
        with open(self.paper_metadata_path, "w") as f:
            json.dump(self.paper_metadata, f, indent=2)

    def preprocess(self, doc, arxiv_id) -> List:
        """
        Clean a document by removing references and splitting into chunks.
        Returns a list of Document objects (chunks).
        """
        content = doc.page_content
        if "References" in content:
            content = content[:content.index("References")]
        doc.page_content = content
        chunks = []
        for c in text_splitter.split_documents([doc]):
            if len(c.page_content) > MIN_CHUNK_LENGTH: # Filter out tiny chunks
                # assign minimal metadata
                c.metadata = {
                "user_id": self.user_id,
                "paper_id": arxiv_id,
                "title": c.metadata.get("Title"),
            }
                chunks.append(c)
        return chunks

    def update_paper_metadata(self, doc_metadata, arxiv_id, len_chunks):
        self.paper_metadata[arxiv_id] = {
                    'Title': doc_metadata.get('Title', 'Unknown'),
                    'Authors': doc_metadata.get('Authors', []),
                    'Published': doc_metadata.get('Published', '')[:4] if doc_metadata.get('Published') else 'Unknown',
                    'Summary': doc_metadata.get('Summary', ''),
                    'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    'total_chunks': len_chunks,
                    'ingested_at': datetime.now().isoformat()  # Track when ingested
                }
        self.save_paper_metadata() 

    def ingest_papers(self, arxiv_ids: List[str]) -> dict:
        """
        Ingest multiple ArXiv papers, one by one.
        Each paper's chunks are added to Qdrant individually, and metadata is updated only if the add succeeds.
        Returns a dict with success/failure info.
        """
        successful = []
        failed = []

        for arxiv_id in arxiv_ids:
            # Paper already in inventory --> Ignore
            if arxiv_id in self.paper_metadata: 
                failed.append({"id": arxiv_id, "reason": "Paper already ingested"})
                continue

            try:
                logging.info(f"📥 Loading paper {arxiv_id} from ArXiv")
                docs = ArxivLoader(query=arxiv_id).load()
                if not docs:
                    failed.append({"id": arxiv_id, "reason": "No content found on ArXiv"})
                    continue

                doc = docs[0]
                chunks = self.preprocess(doc, arxiv_id)

                try:
                    # Add chunks to vector store
                    self.vectorstore.add_documents(chunks)
                    # Only update metadata if add succeeds
                    self.update_paper_metadata(doc.metadata, arxiv_id, len(chunks))
                    successful.append(arxiv_id)
                    logging.info(f"✅ Successfully ingested {arxiv_id}")
                except Exception as e:
                    failed.append({"id": arxiv_id, "reason": f"Vector store update failed: {e}"})

            except Exception as e:
                failed.append({"id": arxiv_id, "reason": f"Ingestion failed: {e}"})

        total = len(successful) + len(failed)
        message = (
            f"✅ Ingested {len(successful)} paper(s), ❌ failed {len(failed)} paper(s)"
            if successful and failed
            else f"✅ Successfully ingested {len(successful)} paper(s)" if successful
            else f"❌ Failed to ingest {len(failed)} paper(s)"
        )
        if failed:
            details = "\n".join([f"• {f['id']}: {f['reason']}" for f in failed])
            message += f"\n\nFailed papers:\n{details}"

        return {"successful": successful, "failed": failed, "total_processed": total, "message": message}

    def get_num_vectors(self) -> int:
        """Return total number of vectors in the collection."""
        try:
            result = self.qdrant_client.count(collection_name=self.collection_name)
            return result.count
        except Exception as e:
            logging.warning(f"Failed to fetch vector count: {e}")
            return 0

    #Note saving
    def save_notes(self, paper_id, text):
        if paper_id not in self.paper_metadata:
            return False
        self.paper_metadata[paper_id]["notes"] = text
        self.save_paper_metadata()  
        return True

    def delete_paper(self, paper_id: str) -> bool:
        """
        Delete all chunks of a given paper from Qdrant and remove its metadata.
        Returns True if deletion succeeded, False otherwise.
        """
        if paper_id not in self.paper_metadata:
            logging.warning(f"Paper {paper_id} not found in metadata.")
            return False
        try:
            # Create a filter to match user_id and paper_id
            delete_filter = models.Filter(
                must=[
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
                    models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id)),
                ]
            )
            # Delete points from vectorstore using filter
            self.vectorstore.client.delete(
                collection_name=self.collection_name,
                points_selector=delete_filter
            )
            # Remove metadata
            del self.paper_metadata[paper_id]
            self.save_paper_metadata()
            logging.info(f"✅ Successfully deleted paper {paper_id} and its chunks.")
            return True
        
        except Exception as e:
            logging.error(f"❌ Failed to delete chunks of {paper_id} from vector store: {e}")
            return False

# === TEST ===
if __name__ == "__main__":
    user_store = PaperVectorStore(user_id="demo_user")
    # Ingest multiple papers at once
    result = user_store.ingest_papers([
        "1706.03762",  # Attention Is All You Need
        "1810.04805",  # BERT
        "2005.11401",  # RAG
        "2512.04148",
        "2509.19391",
        "2303.08774"
    ])
    logging.info(result)
    logging.info(f"Total vectors stored: {user_store.get_num_vectors()}")
    user_store.delete_paper("1810.04805")
  