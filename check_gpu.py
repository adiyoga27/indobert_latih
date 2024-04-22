import torch

if torch.cuda.is_available():
    print('Available GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')