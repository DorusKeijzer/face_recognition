from model import ConvAutoencoder
from dataloader import DataLoader
import torch
import torch.nn.functional as F

CONTRACTIVE_LAMBDA = 0.2

dataloader = DataLoader()
model = ConvAutoencoder()
optimizer = torch.optim.NAdam(model.parameters())

def contractive_loss(input, reconstructed, encoded, lambda_):
    # Reconstruction loss
    mse_loss = F.mse_loss(reconstructed, input)

    # batch_size = input.size(0)
    encoded_sum = encoded.sum()
    grads = torch.autograd.grad(encoded_sum, input, create_graph=True)[0]
    frob_norm = torch.norm(grads, p='fro')  # This gives the Frobenius norm

    return mse_loss + lambda_ * frob_norm

for input_batch, noisy_input in dataloader:
    input_batch = input_batch.requires_grad_()  # so we can take ∂z/∂x
    encoded, reconstructed = model(noisy_input, return_latent=True)  # modify model to return latent
    loss = contractive_loss(input_batch, reconstructed, encoded, CONTRACTIVE_LAMBDA)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


