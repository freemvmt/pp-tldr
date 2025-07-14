import matplotlib.pyplot as plt
import torch


def show_log_mel(log_mel: torch.Tensor, title: str = "Log Mel Spectrogram"):
    plt.figure(figsize=(10, 6))
    plt.imshow(log_mel.numpy(), aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Log Mel Magnitude")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency Bins")
    plt.title(title)
    plt.show()
