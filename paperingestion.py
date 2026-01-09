import os
import json
from datetime import datetime
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from config import BASE_USER_DATA_DIR, COLLECTION_NAME, embedder, qdrant_client
from langchain_community.vectorstores import Qdrant

# TEXT SPLITTER
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " "],
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
        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=embedder,
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
            json.dump(self.paper_metadata, f)

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
            if len(c.page_content) > 200: # Filter out tiny chunks
                # assign minimal metadata
                c.metadata = {
                "user_id": self.user_id,
                "paper_id": arxiv_id,
                "title": c.metadata.get("Title"),
            }
                chunks.append(c)
        return chunks

    def update_paper_metadata(self, doc, arxiv_id, len_chunks):
        self.paper_metadata[arxiv_id] = {
                    'Title': doc.metadata.get('Title', 'Unknown'),
                    'Authors': doc.metadata.get('Authors', []),
                    'Published': doc.metadata.get('Published', '')[:4] if doc.metadata.get('Published') else 'Unknown',
                    'Summary': doc.metadata.get('Summary', ''),
                    'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    'total_chunks': len_chunks,
                    'ingested_at': datetime.now().isoformat()  # Track when ingested
                }
        self.save_paper_metadata() 

    def ingest_papers(self, arxiv_ids: List[str]) -> dict:
        """
        Load multiple ArXiv papers, preprocess, chunk, embed, and add to the vectorstore.
        Returns:
            dict: {
                'successful': list of successfully ingested IDs,
                'failed': list of failed IDs with reasons,
                'total_processed': int,
                'message': str
            }
        """
        successful_ingestions = []
        failed_ingestions = []  # Track failures with reasons
        
        for arxiv_id in arxiv_ids:
            # Paper already in inventory --> Ignore 
            if arxiv_id in self.paper_metadata:
                failed_ingestions.append({'id': arxiv_id, 'reason': 'Paper already ingested'})
                continue
            try:
                print(f"📥 Loading paper {arxiv_id} from ArXiv")
                docs = ArxivLoader(query=arxiv_id).load()
                if not docs:
                    failed_ingestions.append({'id': arxiv_id, 'reason': 'No content found on arXiv'})
                    continue
                    
                doc = docs[0]
                chunks = self.preprocess(doc, arxiv_id)
                print(f"🧩 Generated {len(chunks)} chunks for paper {arxiv_id}")
                    
                try:
                    # Update vectorstore and metadata file
                    self.vectorstore.add_documents(chunks)
                    self.update_paper_metadata(doc, arxiv_id, len(chunks))
                    successful_ingestions.append(arxiv_id)
                except Exception as e:
                    # If saving fails, mark all as failed
                    failed_ingestions.append({'id': arxiv_id, 'reason': f'Vector store update failed: {str(e)}'})                             
         
            except Exception as e:
                error_msg = f"Failed to ingest {arxiv_id}: {str(e)}"
                print(f"❌ {error_msg}")
                failed_ingestions.append({'id': arxiv_id, 'reason': error_msg})
                continue
             
        # Build comprehensive result message
        total_processed = len(successful_ingestions) + len(failed_ingestions)
        
        if successful_ingestions and not failed_ingestions:
            count = len(successful_ingestions)
            message = f"✅ Successfully ingested {count} paper{'s' if count > 1 else ''}!"
        elif successful_ingestions and failed_ingestions:
            success_count = len(successful_ingestions)
            fail_count = len(failed_ingestions)
            message = (
                f"✅ Ingested {success_count} paper{'s' if success_count > 1 else ''}, "
                f"❌ failed {fail_count} paper{'s' if fail_count > 1 else ''}"
            )
        elif not successful_ingestions and failed_ingestions:
            fail_count = len(failed_ingestions)
            message = f"❌ Failed to ingest {fail_count} paper{'s' if fail_count > 1 else ''}"
        else:
            message = "No papers processed"

        # Add details about failures
        if failed_ingestions:
            failure_details = "\n".join([f"• {f['id']}: {f['reason']}" for f in failed_ingestions])
            message += f"\n\nFailed papers:\n{failure_details}"
        
        return {
            'successful': successful_ingestions,
            'failed': failed_ingestions,
            'total_processed': total_processed,
            'message': message
        }

    def get_num_vectors(self) -> int:
        """Return total number of vectors in the collection."""
        try:
            result = self.qdrant_client.count(collection_name=self.collection_name)
            return result.count
        except Exception as e:
            print(f"⚠️ Failed to fetch vector count: {e}")
            return 0

    #Note saving
    def save_notes(self, paper_id, text):
        if paper_id not in self.paper_metadata:
            return False
        self.paper_metadata[paper_id]["notes"] = text
        self.save_paper_metadata()  
        return True

# === TEST ===
if __name__ == "__main__":
    user_store = PaperVectorStore(user_id="demo_user")
    # Ingest multiple papers at once
    user_store.ingest_papers([
        "1706.03762",  # Attention Is All You Need
        "1810.04805",  # BERT
        "2005.11401",  # RAG
    ])
    
    print(f"Total vectors stored: {user_store.get_num_vectors()}")
