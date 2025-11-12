import chromadb
import cv2
import asyncio
import aiohttp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI
from pydantic_ai import Agent, BinaryImage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import get_settings

config = get_settings()

VIDEO_CAPTION_PROMPT = """You are an expert traffic video captioniner. Given a video frame from a vehicle's dashboard camera, generate a concise and descriptive caption that accurately reflects the content of the frame. Focus on key elements such as vehicles, pedestrians, road conditions, traffic signs, and any notable events occurring in the scene. Your caption should be clear and informative, providing a snapshot of the situation depicted in the frame.
"""

class VideoProcessor:
    """
    Ingest and process video files for searching and retrieval.
    Sample NUM_FRAMES_TO_SAMPLE frames from the video for captioning and indexing.
    """
    def __init__(self, video_path: Path):
        self.video_path = video_path
    
        self.model = self._setup_model()
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="video_captions_db")

    def _setup_model(self) -> Agent:
        openai_client = AsyncOpenAI(
            base_url="http://localhost:8080/v1",
            api_key="admin",
        )

        model = OpenAIChatModel("qwen3-vl:32b", provider=OpenAIProvider(openai_client=openai_client))

        agent = Agent(
            model=model,
            system_prompt=VIDEO_CAPTION_PROMPT,
        )

        return agent

    async def generate_caption(self, frame: BinaryImage) -> str:
        """
        Generate a caption for a given video frame using the AI model.

        Args:
            frame (BinaryImage): The video frame to caption.    
        Returns:
            str: The generated caption for the frame.
        """
        response = await self.model.run([frame])
        return response.output  

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for the given text using the embedding model.

        Args:
            text (str): The text to generate an embedding for.
        Returns:
            list[float]: The generated embedding vector.
        """
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {"inputs": [text]}
            async with session.post(
                config.EMBEDDING_MODEL_ENDPOINT,
                data=aiohttp.JsonPayload(payload),
                headers={"Content-Type": "application/json"},
            ) as resp:
                result = await resp.json()
            
            return result[0]

    async def process_video(self):
        """
        Process the video file to sample frames, generate captions, and store them in ChromaDB.
        """
        frames, frame_indices = self._sample_uniform_frames(str(self.video_path), n_frames=config.NUM_FRAMES_TO_SAMPLE)

        for idx, frame in enumerate(frames):
            # Convert the frame (NumPy array) to BinaryImage
            print(self.video_path, idx, frame.shape)
            success, buffer = cv2.imencode(".png", frame)
            if not success:
                print(f"Error encoding frame {idx} to PNG format.")
                continue
            data = BinaryImage(data=buffer.tobytes(), media_type="image/png")

            # Generate caption asynchronously
            caption = await self.generate_caption(data)

            embedding = await self.generate_embedding(caption)

            self.collection.add(
                documents=[caption],
                embeddings=[embedding],
                ids=[f"{self.video_path.stem}_frame_{frame_indices[idx]}"],
                metadatas=[{"video_path": str(self.video_path), "frame_index": int(frame_indices[idx])}],
            )



    def _sample_uniform_frames(self, video_path, n_frames):
        """
        Samples n_frames uniformly from a video.

        Args:
            video_path (str): Path to the video file.
            n_frames (int): The number of frames to sample.

        Returns:
            list: A list of the sampled frames (each as a NumPy array).
        """
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []

        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Handle edge cases
        if total_frames <= 0:
            print("Error: Video has no frames.")
            cap.release()
            return []
        
        if n_frames <= 0:
            n_frames = 1
            
        if n_frames > total_frames:
            print(f"Warning: Requested {n_frames} frames, but video only has {total_frames}.")
            print("Returning all frames.")
            n_frames = total_frames

        # 1. Calculate the indices of the frames to sample
        # We use np.linspace to get n_frames evenly spaced points between
        # 0 (the first frame) and total_frames - 1 (the last frame).
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

        # 2. Use a set for fast O(1) lookups
        target_indices = set(indices)
        
        sampled_frames = []
        frame_counter = 0

        # 3. Loop through the video and grab the target frames
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Stop if we're at the end of the video
            if not ret:
                break

            # Check if the current frame is one we want
            if frame_counter in target_indices:
                sampled_frames.append(frame)
                
                # Optimization: If we've found all our frames, we can stop early
                if len(sampled_frames) == n_frames:
                    break

            frame_counter += 1

        cap.release()
        
        print(f"Successfully sampled {len(sampled_frames)} frames at indices: {indices}")
        return sampled_frames, indices
    

async def main():
    paths = [
        Path("/home/aki/workspace/learning/zaloai/zaloai/traffic_buddy_train+public_test/train/videos"),
        Path("/home/aki/workspace/learning/zaloai/zaloai/traffic_buddy_train+public_test/public_test/videos")
    ]
    video_files = []
    for path in paths:
        video_files.extend(list(path.glob("*.mp4")))

    for video_path in tqdm(video_files):
        print(video_path)
        # processor = VideoProcessor(video_path)
        # await processor.process_video()

if __name__ == "__main__":
    asyncio.run(main())