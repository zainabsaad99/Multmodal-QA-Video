# data_preparation.py

import os
import json
import cv2
from tqdm import tqdm
import yt_dlp
import whisper

class VideoDataProcessor:
    def __init__(self):
        # Initialize paths and directories
        self.video_path = "temp_video.mp4"  # Temporary video file path
        self.transcript_path = "processed_data/transcription.json" # Temporary transcription file path
        self.frames_dir = "processed_data/frames" # Directory to save frames

    # Initialize Whisper model  
    def fetch_video_from_youtube(self, url):
        """Download video from YouTube using yt-dlp"""
        download_options = {
            'outtmpl': self.video_path, # Output file name
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',  # Force MP4 format
            'quiet': True, # Suppress output
            'progress_hooks': [self._download_progress_hook] # Hook for download progress
        }
        
        # Create the directory if it doesn't exist
        with yt_dlp.YoutubeDL(download_options) as downloader:
            downloader.download([url])

    # Hook for download progress
    # This function is called by yt-dlp to report download progress     
    def _download_progress_hook(self, d):
        if d['status'] == 'downloading':
            print(f"Downloading... {d['_percent_str']} complete", end='\r')

    # Hook for download completion
    # This function is called by yt-dlp when the download is complete
    def generate_transcription(self):
        """Convert audio to text using Whisper"""
        if not os.path.exists(self.video_path):
            # Try with .mkv extension if .mp4 not found
            mkv_path = self.video_path + '.mkv' # Change to .mkv extension
            if os.path.exists(mkv_path): # Check if .mkv file exists
                self.video_path = mkv_path # Update video path to .mkv
            else:
                raise FileNotFoundError(f"Video file not found at {self.video_path} or {mkv_path}")
            
        print("\nInitializing Whisper model...")
        # Load the Whisper model
        speech_model = whisper.load_model("small")
        transcription = speech_model.transcribe(self.video_path)
        
        # Save the transcription to a JSON file
        os.makedirs(os.path.dirname(self.transcript_path), exist_ok=True)
        with open(self.transcript_path, "w") as output_file:
            json.dump(transcription['segments'], output_file, indent=2)
            
        return transcription['segments']

    # Capture video frames at specified intervals
    # This function extracts frames from the video at specified intervals
    def capture_video_frames(self, interval_seconds=5):
        """Extract frames at specified intervals"""
        if not os.path.exists(self.video_path):
            # Try with .mkv extension if .mp4 not found
            mkv_path = self.video_path + '.mkv'
            if os.path.exists(mkv_path):
                self.video_path = mkv_path
            else:
                raise FileNotFoundError(f"Video file not found at {self.video_path} or {mkv_path}")
        # Open the video file
        video = cv2.VideoCapture(self.video_path)
        # Check if video opened successfully
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        frame_step = int(frames_per_second * interval_seconds)
        
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Create the directory if it doesn't exist
        current_frame = 0
        saved_count = 0
        # Initialize saved_count to 0
        # Loop through the video frames
        print("Extracting key frames...")
        while video.isOpened():
            success, frame_image = video.read()
            if not success:
                break
            # Check if the current frame is a key frame 
            if current_frame % frame_step == 0:
                output_path = os.path.join(
                    self.frames_dir, 
                    f"keyframe_{saved_count:05d}.png"
                )
                cv2.imwrite(output_path, frame_image)
                saved_count += 1
                
            current_frame += 1
            
        video.release()
        print(f"Saved {saved_count} frames to {self.frames_dir}")

def main():
    # Initialize the video data processor
    processor = VideoDataProcessor()
    youtube_link = "https://www.youtube.com/watch?v=dARr3lGKwk8"
    
    try:
        # Fetch video, generate transcription, and capture frames
        # The following functions are called in sequence to process the video
        print("Starting video processing pipeline")
        # Download video from YouTube
        processor.fetch_video_from_youtube(youtube_link)
        # Generate transcription using Whisper
        processor.generate_transcription()
        # Capture video frames at specified intervals
        processor.capture_video_frames()
        # Print success message
        print("Processing completed successfully")
    except Exception as error:
        print(f"Error occurred: {str(error)}")

if __name__ == "__main__":
    main()