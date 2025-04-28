# retrieval_system.py

import os
import json
import pickle
import numpy as np
import faiss
import psycopg2
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

class RetrievalSystemBuilder:
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
                'text_embeddings': "embeddings/text_embeddings.npy",
                'image_embeddings': "embeddings/image_embeddings.npy",
                'transcript': "processed_data/transcription.json",
                'output': {
                    'faiss_text': "retrieval/faiss_text.index",
                    'faiss_image': "retrieval/faiss_image.index",
                    'tfidf': "retrieval/tfidf_vectorizer.pkl",
                    'bm25': "retrieval/bm25_model.pkl"
                }
            }
        }
        
        # Create output directory if it doesn't exist
        os.makedirs("retrieval", exist_ok=True)

    # Initialize the database connection
    def _get_db_connection(self):
        """Establish and return a PostgreSQL database connection"""
        return psycopg2.connect(
            dbname=self.config['database']['dbname'],
            user=self.config['database']['user'],
            password=self.config['database']['password'],
            host=self.config['database']['host'],
            port=self.config['database']['port']
        )

    # Create the table for text embeddings if needed
    def _load_transcript_data(self) -> List[str]:
        """Load and validate transcript data"""
        if not os.path.exists(self.config['paths']['transcript']):
            raise FileNotFoundError(f"Transcript file not found at {self.config['paths']['transcript']}")
        
        with open(self.config['paths']['transcript'], 'r') as file:
            segments = json.load(file)
        
        if not isinstance(segments, list) or len(segments) == 0:
            raise ValueError("Invalid transcript data format")
            
        return [seg['text'] for seg in segments]

    # Create the table for text embeddings if needed
    # Create the table for image embeddings if needed
    def _load_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load both text and image embeddings"""
        if not os.path.exists(self.config['paths']['text_embeddings']):
            raise FileNotFoundError(f"Text embeddings not found at {self.config['paths']['text_embeddings']}")
        if not os.path.exists(self.config['paths']['image_embeddings']):
            raise FileNotFoundError(f"Image embeddings not found at {self.config['paths']['image_embeddings']}")
        #   Load the text and image embeddings
        #   Ensure the embeddings are in the correct format (numpy arrays)  
        text_embeddings = np.load(self.config['paths']['text_embeddings'])
        image_embeddings = np.load(self.config['paths']['image_embeddings'])
        
        return text_embeddings, image_embeddings

    # Create the table for text embeddings if needed
    # Create the table for image embeddings if needed
    def _setup_database_tables(self) -> None:
        """Initialize the database schema if it doesn't exist"""
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Ensure vector extension is enabled
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create main table for text embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_embeddings (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector(384)  -- Dimension must match your embedding size
                );
            """)
            
            conn.commit()
            print("Database tables initialized successfully")
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise RuntimeError(f"Failed to initialize database tables: {str(e)}")
        finally:
            if conn:
                cursor.close()
                conn.close()

    # Create the table for image embeddings if needed
    def _store_embeddings_in_database(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """Store text embeddings in PostgreSQL database"""
        conn = None
        try:
            # Establish a connection to the database
            conn = self._get_db_connection()
            cursor = conn.cursor()
            ## Ensure the table exists
            print("Storing text embeddings in database...")
            for text, embedding in zip(texts, embeddings):
                # Convert numpy array to PostgreSQL vector format
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                cursor.execute(
                    "INSERT INTO text_embeddings (text, embedding) VALUES (%s, %s);",
                    (text, embedding_str)
                )
            # Commit the transaction
            conn.commit()
            print(f"Successfully stored {len(texts)} embeddings in database")
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise RuntimeError(f"Failed to store embeddings in database: {str(e)}")
        finally:
            if conn:
                cursor.close()
                conn.close()

    # Create the table for image embeddings if needed
    def _build_database_indexes(self) -> None:
        """Create optimized indexes for vector search in PostgreSQL"""
        conn = None
        try:
            # Establish a connection to the database
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            print("Building database indexes for vector search...")
            
            # Create IVFFLAT index with vector_l2_ops operator class
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS ivfflat_index
                ON text_embeddings USING ivfflat (embedding vector_l2_ops)
                WITH (lists = 100);  -- Adjust lists based on dataset size
            """)
            
            # Create HNSW index with proper parameters
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS hnsw_index
                ON text_embeddings USING hnsw (embedding vector_l2_ops)
                WITH (m = 16, ef_construction = 64);  -- Good default values
            """)
            # Commit the transaction
            conn.commit()
            print("Database indexes created successfully")
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise RuntimeError(f"Failed to create database indexes: {str(e)}")
        finally:
            if conn:
                cursor.close()
                conn.close()

    # Create the table for image embeddings if needed
    def _build_faiss_indexes(self, text_embeddings: np.ndarray, image_embeddings: np.ndarray) -> None:
        """Create FAISS indexes for both text and image embeddings"""
        try:
            print("Building FAISS index for text embeddings...")
            text_index = faiss.IndexFlatL2(text_embeddings.shape[1])
            text_index.add(text_embeddings)
            faiss.write_index(text_index, self.config['paths']['output']['faiss_text'])
            
            print("Building FAISS index for image embeddings...")
            image_index = faiss.IndexFlatL2(image_embeddings.shape[1])
            image_index.add(image_embeddings)
            faiss.write_index(image_index, self.config['paths']['output']['faiss_image'])
            
            print("FAISS indexes built and saved successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to build FAISS indexes: {str(e)}")
        
    # Create the table for image embeddings if needed
    def _build_lexical_models(self, texts: List[str]) -> None:
        """Create TF-IDF and BM25 models for lexical search"""
        try:
            print("Building TF-IDF model...")
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            with open(self.config['paths']['output']['tfidf'], "wb") as f:
                pickle.dump((vectorizer, tfidf_matrix), f)
            
            print("Building BM25 model...")
            tokenized_corpus = [text.lower().split() for text in texts]
            bm25 = BM25Okapi(tokenized_corpus)
            with open(self.config['paths']['output']['bm25'], "wb") as f:
                pickle.dump(bm25, f)
            
            print("Lexical search models built successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to build lexical search models: {str(e)}")
        
    # Create the table for image embeddings if needed
    def build_retrieval_system(self) -> None:
        """Execute the full retrieval system build pipeline"""
        try:
            print("Starting retrieval system build process...")
            
            # Load required data
            texts = self._load_transcript_data()
            text_embeddings, image_embeddings = self._load_embeddings()
            
            # Build semantic retrieval components
            self._setup_database_tables()
            self._store_embeddings_in_database(text_embeddings, texts)
            self._build_database_indexes()
            self._build_faiss_indexes(text_embeddings, image_embeddings)
            
            # Build lexical retrieval components
            self._build_lexical_models(texts)
            
            print("\nRetrieval system built successfully!")
            print(f"FAISS indexes saved to: {self.config['paths']['output']['faiss_text']} and {self.config['paths']['output']['faiss_image']}")
            print(f"Lexical models saved to: {self.config['paths']['output']['tfidf']} and {self.config['paths']['output']['bm25']}")
            
        except Exception as e:
            print(f"\nError building retrieval system: {str(e)}")
            raise

if __name__ == "__main__":
    builder = RetrievalSystemBuilder()
    builder.build_retrieval_system()