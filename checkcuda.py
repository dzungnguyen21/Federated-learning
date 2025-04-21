import torch

print(torch.cuda.is_available())  # Should return True if CUDA is available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # Should print "cuda" if GPU is available
print(torch.__version__)  # PyTorch version
print(torch.version.cuda)  # CUDA version used by PyTorch
print(torch.backends.cudnn.version())  # cuDNN version
