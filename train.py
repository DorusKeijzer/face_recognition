from model import ConvAutoencoder
from dataloader import train_dataloader
import torch
import torch.nn.functional as F

CONTRACTIVE_LAMBDA = 0.2
EPOCHS = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = ConvAutoencoder().to(device)
optimizer = torch.optim.NAdam(model.parameters())

def contractive_loss(target, reconstructed, encoded, input_tensor, lambda_):
    mse_loss = F.mse_loss(reconstructed, target)
    encoded_sum = encoded.sum()
    grads = torch.autograd.grad(encoded_sum, input_tensor, create_graph=True)[0]
    frob_norm = torch.norm(grads, p='fro')
    return mse_loss + lambda_ * frob_norm





def train_loop():
    for batch, (input_batch, noisy_input) in enumerate(train_dataloader):
        input_batch = input_batch.float().to(device).requires_grad_()  # so we can take ∂z/∂x
        noisy_input = noisy_input.float().to(device).requires_grad_()
        encoded, reconstructed = model(noisy_input)  
        loss = contractive_loss(input_batch, reconstructed, encoded, noisy_input, CONTRACTIVE_LAMBDA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % int((len(train_dataloader)/10)) == 0:
            print(f"\tBatch: {batch}, Loss: {loss.item()}")
        

if __name__ == "__main__":
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}")
        train_loop()

