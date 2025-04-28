# embedding_generator.py

import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
# from data_preparation import VideoDataProcessor
# from evaluation import evaluate_retrieval
class EmbeddingGenerator:
    def __init__(self):
        self.config = {
            'text_model': "sentence-transformers/all-MiniLM-L6-v2",
            'image_model': "openai/clip-vit-base-patch32",
            'data_paths': {
                'transcript': "processed_data/transcription.json",
                'frames': "processed_data/frames/",
                'output': {
                    'text': "embeddings/text_embeddings.npy",
                    'image': "embeddings/image_embeddings.npy"
                }
            }
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize the VideoDataProcessor class
    # self.video_processor = VideoDataProcessor()
    def load_transcription_data(self):
        """Load and validate transcription data"""
        # Check if the transcription file exists
        if not os.path.exists(self.config['data_paths']['transcript']):
            raise FileNotFoundError(f"Transcript file not found at {self.config['data_paths']['transcript']}")
        # Load the transcription data
        with open(self.config['data_paths']['transcript'], 'r') as file:
            data = json.load(file)
        # Validate the data format
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Invalid transcript data format")
        return data
    
    # Initialize the VideoDataProcessor class
    def create_text_embeddings(self, text_segments):
        """Generate embeddings for text segments"""
        # Check if the text segments are valid
        model = SentenceTransformer(self.config['text_model'])
        # Check if the text segments are valid
        texts = [segment['text'] for segment in text_segments]
        
        print("Generating text embeddings...")
        # Generate embeddings using the SentenceTransformer model
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    # Initialize the VideoDataProcessor class
    def create_image_embeddings(self):

        """Generate embeddings for video frames"""
        if not os.path.exists(self.config['data_paths']['frames']):
            raise FileNotFoundError(f"Frames directory not found at {self.config['data_paths']['frames']}")
       
        # Check if the frames directory is empty    
        processor = CLIPProcessor.from_pretrained(self.config['image_model'])

        # Load the CLIP model
        model = CLIPModel.from_pretrained(self.config['image_model']).to(self.device)
        # Check if the frames directory is empty
        frame_files = sorted([
            f for f in os.listdir(self.config['data_paths']['frames'])
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if not frame_files:
            raise ValueError("No valid image files found in frames directory")
        
        embeddings = []
        print("Processing video frames...")
        
        # Process each frame and generate embeddings
        for frame_file in tqdm(frame_files, desc="Generating image embeddings"):
            try:
                image_path = os.path.join(self.config['data_paths']['frames'], frame_file)
                image = Image.open(image_path).convert("RGB")
                
                inputs = processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                
                features = model.get_image_features(**inputs)
                embeddings.append(features.cpu().detach().numpy()[0])
            except Exception as e:
                print(f"\nWarning: Could not process {frame_file}: {str(e)}")
                continue
                
        return np.array(embeddings)

    # Initialize the VideoDataProcessor class
    def save_embeddings(self, embeddings, output_type):
        """Save embeddings to disk"""
        os.makedirs(os.path.dirname(self.config['data_paths']['output'][output_type]), exist_ok=True)
        np.save(self.config['data_paths']['output'][output_type], embeddings)

    # Initialize the VideoDataProcessor class
    def run_pipeline(self):
        """Execute the full embedding generation process"""
        try:
            print("Starting embedding generation pipeline")
            
            # Load and process text data
            transcript_data = self.load_transcription_data()
            text_embeddings = self.create_text_embeddings(transcript_data)
            self.save_embeddings(text_embeddings, 'text')
            
            # Process image data
            image_embeddings = self.create_image_embeddings()
            self.save_embeddings(image_embeddings, 'image')
            
            print("\nEmbedding generation completed successfully")
            print(f"Text embeddings saved to: {self.config['data_paths']['output']['text']}")
            print(f"Image embeddings saved to: {self.config['data_paths']['output']['image']}")
            
        except Exception as e:
            print(f"\nError in embedding generation: {str(e)}")
            raise

if __name__ == "__main__":
    generator = EmbeddingGenerator()
    generator.run_pipeline()