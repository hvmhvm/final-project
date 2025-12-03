import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# =========================
# Data utilities
# =========================

DNA = ['A','C','G','T']
vocab = {c:i for i,c in enumerate(DNA)}

def encode(seq):
    x = torch.zeros(len(seq), 4)
    for i,c in enumerate(seq):
        x[i][vocab[c]] = 1
    return x

def decode(tensor):
    idxs = tensor.argmax(dim=1)
    return ''.join(DNA[i] for i in idxs)

def generate_clean_seq(L=60):
    return ''.join(random.choice(DNA) for _ in range(L))

def add_noise(seq, rate=0.1):
    seq = list(seq)
    for i in range(len(seq)):
        if random.random() < rate:
            seq[i] = random.choice(DNA)
    return ''.join(seq)

def generate_pair(L=60, noise=0.1):
    clean = generate_clean_seq(L)
    noisy = add_noise(clean, noise)
    return encode(noisy), encode(clean), clean, noisy

# =========================
# Model
# =========================

class DNA_AE(nn.Module):
    def __init__(self, L=60):
        super().__init__()
        self.L = L
        self.encoder = nn.Sequential(
            nn.Linear(4*L, 128),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(128, 4*L)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(-1, self.L, 4)
        out = torch.softmax(out, dim=2)
        return out


# =========================
# Train model
# =========================

L = 60
model = DNA_AE(L=L)
opt = optim.Adam(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()

steps = 2000

for step in range(steps):
    noisy, clean, _, _ = generate_pair(L)

    noisy = noisy.flatten().unsqueeze(0)
    clean_labels = clean.argmax(dim=1).unsqueeze(0)

    opt.zero_grad()
    pred = model(noisy)

    loss = ce(pred.view(-1,4), clean_labels.view(-1))
    loss.backward()
    opt.step()

    if step % 200 == 0:
        print(f"Step {step}, Loss = {loss.item():.4f}")


# =========================
# Run 100 tests and print each sequence
# =========================

accuracies = []
position_correct = torch.zeros(L)
num_tests = 100

print("\n====================")
print("  TEST SEQUENCES")
print("====================\n")

for t in range(num_tests):
    noisy, clean, clean_str, noisy_str = generate_pair(L)
    x_test = noisy.flatten().unsqueeze(0)

    pred = model(x_test).squeeze(0)
    reconstructed = decode(pred)

    # compute accuracy
    correct = sum(1 for a,b in zip(clean_str, reconstructed) if a == b)
    acc = correct / L
    accuracies.append(acc)

    # record per-position accuracy
    for i in range(L):
        if clean_str[i] == reconstructed[i]:
            position_correct[i] += 1

    # PRINT EACH TEST SEQUENCE
    print(f"[Sample {t+1}] Accuracy = {acc:.3f}")
    print("Clean:        ", clean_str)
    print("Noisy:        ", noisy_str)
    print("Reconstructed:", reconstructed)
    print("-" * 80)

position_accuracy = position_correct / num_tests
print("\nAverage accuracy over 100 tests:", sum(accuracies)/len(accuracies))


# =========================
# Plot results
# =========================

plt.figure()
plt.hist(accuracies, bins=10)
plt.title("Reconstruction Accuracy Distribution (100 samples)")
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.savefig("accuracy_hist.png", dpi=300)

plt.figure()
plt.plot(position_accuracy)
plt.title("Per-Position Reconstruction Accuracy")
plt.xlabel("Position")
plt.ylabel("Accuracy")
plt.savefig("position_accuracy.png", dpi=300)

print("\nPlots saved: accuracy_hist.png , position_accuracy.png")


