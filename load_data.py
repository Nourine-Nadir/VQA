import cv2
import torch
import numpy as np
from torchvision import transforms
from play_video import display, play_tensor_video

def load_video_frames(video_path, num_frames=8, img_size=(224,224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select `num_frames` evenly spaced frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(transform(frame))

    cap.release()

    if len(frames) < num_frames:
        # If the video is too short, pad with the last frame
        while len(frames) < num_frames:
            frames.append(frames[-1])


    frames = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)
    return frames


if __name__ == "__main__":

# Example usage
    video_tensor = load_video_frames("/media/nadir/SSD/VQA Data/KoNVID/KoNViD_1k_videos/3425371223.mp4", num_frames=25)
    play_tensor_video(video_tensor, fps=25)
    video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension: (B, C, T, H, W)

    print(video_tensor.shape)  # Should print: torch.Size([1, 3, 8, 224, 224])