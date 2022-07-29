import torch
import torch.nn as nn
import torch.nn.functional as F

# loss = nn.L1Loss()
# input1 = torch.randn(3, 2, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(2)
# print(input1)
# print(target)
#
# output = F.cross_entropy(input1, target, reduction='none')
#
#
# # output2 = F.softmax(input1)
# output2 = F.log_softmax(input1)
# output3 = F.nll_loss(output2, target, reduction='none')
# print(output)
# print(output4)


T = 8      # Input sequence length
C = 3      # Number of classes (including blank)
N = 2      # Batch size
S = 7      # Target sequence length of longest target in batch (padding length)
S_min = 4  # Minimum target length, for demonstration purposes

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()

# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

print(input)
print(input.size())
print(target)
print(target.size())

input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)

print(input_lengths)

print(target_lengths)


ctc_loss = nn.CTCLoss(reduction='none')
loss = ctc_loss(input, target, input_lengths, target_lengths)
print(loss)
# loss.backward()