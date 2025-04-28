# semantic_retriever.py

import os
import json
import numpy as np
import faiss
import pickle
import psycopg2
from text_processing import TextVectorizer
from vector_db import VectorDatabaseManager

# This script builds a retrieval system for video data using semantic and lexical methods.
class RetrievalSystemBuilder:
    def __init__(self, config):
        self.config = config
        self.vector_dimensions = {
            'text': 384,
            'image': 512
        }
        
    def initialize_database(self):
        """Set up PostgreSQL with vector extension"""
        db_manager = VectorDatabaseManager(
            host=self.config['pg_host'],
            port=self.config['pg_port'],
            dbname=self.config['pg_dbname'],
            user=self.config['pg_user'],
            password=self.config['pg_password']
        )
        db_manager.initialize_vector_tables()
        return db_manager
    
    def build_semantic_indexes(self, embeddings, db_manager):
        """Create FAISS and pgvector indexes"""
        # FAISS index construction
        for modality in ['text', 'image']:
            index = faiss.IndexFlatL2(self.vector_dimensions[modality])
            index.add(embeddings[modality])
            faiss.write_index(index, self.config[f'faiss_{modality}_index'])
        
        # PostgreSQL vector indexes
        db_manager.create_ivfflat_index(lists=100)
        db_manager.create_hnsw_index(m=16, ef_construction=64)

    def build_lexical_models(self, texts):
        """Create traditional text retrieval models"""
        vectorizer = TextVectorizer()
        vectorizer.train_tfidf(texts)
        vectorizer.train_bm25(texts)
        vectorizer.save_models(
            tfidf_path=self.config['tfidf_model'],
            bm25_path=self.config['bm25_model']
        )

    def process_data(self):
        """Main execution pipeline"""
        # Load and prepare data
        with open(self.config['transcript_path'], 'r') as f:
            segments = json.load(f)
        
        processed_data = {
            'texts': [seg['text'] for seg in segments],
            'embeddings': {
                'text': np.load(self.config['text_embeddings']),
                'image': np.load(self.config['image_embeddings'])
            }
        }

        # Build retrieval systems
        db_manager = self.initialize_database()
        self.build_semantic_indexes(processed_data['embeddings'], db_manager)
        self.build_lexical_models(processed_data['texts'])

        # Store in database
        db_manager.insert_embeddings(
            texts=processed_data['texts'],
            embeddings=processed_data['embeddings']['text']
        )

# Initialize the database connection and vector tables
class VectorDatabaseManager:
    """Handles all PostgreSQL vector operations"""
    # This class manages the connection to the PostgreSQL database and provides methods for creating vector tables, inserting embeddings, and creating indexes.
    def __init__(self, host, port, dbname, user, password):
        self.connection_params = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password
        }
    def _get_connection(self):
        return psycopg2.connect(**self.connection_params)
    def initialize_vector_tables(self):
        """Create tables with vector support"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_vectors (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        vector vector(384)
                    );
                """)
            conn.commit()
    def create_ivfflat_index(self, lists=100):
        """Create IVFFLAT index for approximate search"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS ivfflat_idx 
                    ON document_vectors USING ivfflat (vector vector_l2_ops)
                    WITH (lists = {lists});
                """)
            conn.commit()
    def create_hnsw_index(self, m=16, ef_construction=64):
        """Create HNSW index for graph-based search"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS hnsw_idx
                    ON document_vectors USING hnsw (vector vector_l2_ops)
                    WITH (m = {m}, ef_construction = {ef_construction});
                """)
            conn.commit()

    def insert_embeddings(self, texts, embeddings):
        """Batch insert embeddings with their text"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                for text, embedding in zip(texts, embeddings):
                    cursor.execute(
                        "INSERT INTO document_vectors (content, vector) VALUES (%s, %s);",
                        (text, list(embedding))
                    )
            conn.commit()

# Initialize the database connection and vector tables
class TextVectorizer:
    """Handles traditional text vectorization methods"""
    def __init__(self):
        self.tfidf = None
        self.bm25 = None

    def train_tfidf(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
    
    def train_bm25(self, texts):
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
    

    def save_models(self, tfidf_path, bm25_path):
        with open(tfidf_path, 'wb') as f:
            pickle.dump((self.tfidf, self.tfidf_matrix), f)
        
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)

if __name__ == "__main__":
    config = {
        'pg_host': "localhost",
        'pg_port': 5432,
        'pg_dbname': "RAG-Video",
        'pg_user': "postgres",
        'pg_password': "zainab",
        'transcript_path': "data/transcription.json",
        'text_embeddings': "embeddings/text_embedding.npy",
        'image_embeddings': "embeddings/image_embeddings.npy",
        'faiss_text_index': "indexes/text_index.faiss",
        'faiss_image_index': "indexes/image_index.faiss",
        'tfidf_model': "models/tfidf.pkl",
        'bm25_model': "models/bm25.pkl"
    }

    os.makedirs("indexes", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    builder = RetrievalSystemBuilder(config)
    builder.process_data()
    print("Retrieval system construction completed successfully")