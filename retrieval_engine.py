# retrieval_engine.py

import os
import json
import pickle
import numpy as np
import faiss
import psycopg2
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

class RetrievalEngine:
    def __init__(self):
        self.config = {
            'database': {
                'host': "localhost",
                'port': 5432,
                'dbname': "RAG-Video",
                'user': "postgres",
                'password': "zainab"
            },
            'paths': {
                'transcript': "processed_data/transcription.json",
                'faiss_text_index': "retrieval/faiss_text.index",
                'tfidf_model': "retrieval/tfidf_vectorizer.pkl",
                'bm25_model': "retrieval/bm25_model.pkl"
            },
            'models': {
                'text_encoder': "sentence-transformers/all-MiniLM-L6-v2"
            }
        }
        
        # Load data and initialize components
        self._load_data()
        self.text_encoder = SentenceTransformer(self.config['models']['text_encoder'])

    # Load transcript data and validate structure   
    def _load_data(self) -> None:
        """Load transcript data and validate structure"""
        if not os.path.exists(self.config['paths']['transcript']):
            raise FileNotFoundError(f"Transcript file not found at {self.config['paths']['transcript']}")
        # Load the transcript data
        with open(self.config['paths']['transcript'], 'r') as f:
            self.segments = json.load(f)
        # Validate the structure of the transcript data
        if not isinstance(self.segments, list):
            raise ValueError("Transcript data should be a list of segments")
            
        self.texts = [seg['text'] for seg in self.segments]
        self.timestamps = [seg.get('start', 0) for seg in self.segments]  # Default to 0 if no timestamp
   
    # Initialize the VideoDataProcessor class
    def _get_db_connection(self):
        """Establish and return a PostgreSQL database connection"""
        return psycopg2.connect(
            dbname=self.config['database']['dbname'],
            user=self.config['database']['user'],
            password=self.config['database']['password'],
            host=self.config['database']['host'],
            port=self.config['database']['port']
        )

    # Build FAISS index for semantic search
    def semantic_search_faiss(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Perform semantic search using FAISS index
        
        Args:
            query: The search query string
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with text and timestamp
        """
        # Check if FAISS index exists
        if not os.path.exists(self.config['paths']['faiss_text_index']):
            raise FileNotFoundError("FAISS index not found. Please build the retrieval system first.")
            
        try:
            # Load the FAISS index
            index = faiss.read_index(self.config['paths']['faiss_text_index'])
            # Encode the query using the text encoder
            # Check if the index is trained
            query_vec = self.text_encoder.encode([query]).astype(np.float32)
            # Check if the index is trained
            distances, indices = index.search(query_vec, top_k)
            # Check if the index is trained
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.segments):
                    results.append({
                        "text": self.segments[idx]['text'],
                        "timestamp": self.segments[idx].get('start', 0)
                    })
            return results
            
        except Exception as e:
            raise RuntimeError(f"FAISS search failed: {str(e)}")

    # Build PostgreSQL vector index for semantic search
    def semantic_search_pgvector(self, query: str, method: str = "ivfflat", top_k: int = 3) -> List[Dict]:
        """
        Perform semantic search using PostgreSQL vector indexes
        
        Args:
            query: The search query string
            method: Index method to use ('ivfflat' or 'hnsw')
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with text and timestamp
        """
        # Check if the method is valid
        if method not in ["ivfflat", "hnsw"]:
            raise ValueError("Invalid method. Choose 'ivfflat' or 'hnsw'.")

        # Check if PostgreSQL vector index exists   
        conn = None
        try:
            # Check if the database connection is valid
            query_vec = self.text_encoder.encode([query])[0]
            embedding_str = "[" + ",".join(map(str, query_vec)) + "]"
            # Check if the embedding string is valid
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Search using <-> distance operator
            search_query = f"""
                SELECT text, id FROM text_embeddings
                ORDER BY embedding <-> '{embedding_str}'
                LIMIT {top_k};
            """
            # Execute the search query
            cursor.execute(search_query)
            rows = cursor.fetchall()
            # Check if the rows are valid
            results = []
            for text, idx in rows:
                if 0 <= idx-1 < len(self.segments):
                    results.append({
                        "text": text,
                        "timestamp": self.segments[idx-1].get('start', 0)
                    })
                else:
                    results.append({
                        "text": text,
                        "timestamp": 0
                    })
            return results
            
        except Exception as e:
            raise RuntimeError(f"PostgreSQL vector search failed: {str(e)}")
        finally:
            if conn:
                conn.close()

    # Build TF-IDF model for lexical search
    def lexical_search_tfidf(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Perform lexical search using TF-IDF
        
        Args:
            query: The search query string
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with text and timestamp
        """
        # Check if TF-IDF model exists
        if not os.path.exists(self.config['paths']['tfidf_model']):
            raise FileNotFoundError("TF-IDF model not found. Please build the retrieval system first.")
            
        try:
            with open(self.config['paths']['tfidf_model'], "rb") as f:
                vectorizer, tfidf_matrix = pickle.load(f)
                
            query_vec = vectorizer.transform([query])
            scores = (tfidf_matrix @ query_vec.T).toarray().squeeze()
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if 0 <= idx < len(self.segments):
                    results.append({
                        "text": self.segments[idx]['text'],
                        "timestamp": self.segments[idx].get('start', 0)
                    })
            return results
            
        except Exception as e:
            raise RuntimeError(f"TF-IDF search failed: {str(e)}")
        
    # Build BM25 model for lexical search
    def lexical_search_bm25(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Perform lexical search using BM25
        
        Args:
            query: The search query string
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with text and timestamp
        """
        if not os.path.exists(self.config['paths']['bm25_model']):
            raise FileNotFoundError("BM25 model not found. Please build the retrieval system first.")
            
        try:
            with open(self.config['paths']['bm25_model'], "rb") as f:
                bm25 = pickle.load(f)
            # Check if the BM25 model is valid
            # Tokenize the query
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            # Check if the top indices are valid
            results = []
            for idx in top_indices:
                if 0 <= idx < len(self.segments):
                    results.append({
                        "text": self.segments[idx]['text'],
                        "timestamp": self.segments[idx].get('start', 0)
                    })
            return results
            
        except Exception as e:
            raise RuntimeError(f"BM25 search failed: {str(e)}")
        
    # Build hybrid search combining semantic and lexical methods
    # This function combines semantic and lexical search methods to improve retrieval accuracy
    def hybrid_search(self, query: str, top_k: int = 3, weights: tuple = (0.5, 0.5)) -> List[Dict]:
        """
        Perform hybrid search combining semantic and lexical methods
        
        Args:
            query: The search query string
            top_k: Number of results to return
            weights: Tuple of (semantic_weight, lexical_weight)
            
        Returns:
            List of result dictionaries with text and timestamp
        """
        try:
            # Get semantic results (FAISS)
            semantic_results = self.semantic_search_faiss(query, top_k*2)
            
            # Get lexical results (BM25)
            lexical_results = self.lexical_search_bm25(query, top_k*2)
            
            # Combine and rerank results
            combined = {}
            for idx, result in enumerate(semantic_results):
                combined[result['text']] = combined.get(result['text'], 0) + weights[0] * (1 - idx/(top_k*2))
                
            for idx, result in enumerate(lexical_results):
                combined[result['text']] = combined.get(result['text'], 0) + weights[1] * (1 - idx/(top_k*2))
                
            # Sort by combined score and return top_k
            sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # Return in standard format with timestamps
            final_results = []
            for text, score in sorted_results:
                for seg in self.segments:
                    if seg['text'] == text:
                        final_results.append({
                            "text": text,
                            "timestamp": seg.get('start', 0),
                            "score": score
                        })
                        break
            return final_results
            
        except Exception as e:
            raise RuntimeError(f"Hybrid search failed: {str(e)}")

if __name__ == "__main__":
    # Example usage
    engine = RetrievalEngine()
    
    query = "what is the professor name?"
    print("\nFAISS Results:")
    print(engine.semantic_search_faiss(query))
    
    print("\nPostgreSQL IVFFlat Results:")
    print(engine.semantic_search_pgvector(query, method="ivfflat"))
    
    print("\nTF-IDF Results:")
    print(engine.lexical_search_tfidf(query))
    
    print("\nBM25 Results:")
    print(engine.lexical_search_bm25(query))
    
    print("\nHybrid Results:")
    print(engine.hybrid_search(query))