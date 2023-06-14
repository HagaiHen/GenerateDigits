import torch

def get_device():
    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

batch_size = 32
lr = 0.0001
num_epochs = 100

hidden_dim_1 = 256
hidden_dim_2 = 512
hidden_dim_3 = 1024
