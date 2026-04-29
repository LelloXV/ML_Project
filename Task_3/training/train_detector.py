# training/train_detector.py
import torch
from torch.utils.data import DataLoader

def train_detector(model, dataset, epochs=30, lr=1e-4, device="cuda"):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(dataset, batch_size=16, shuffle=True,
                        collate_fn=lambda b: list(zip(*b)))
    for epoch in range(epochs):
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)         # torchvision returns dict
            loss = sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        print(f"epoch {epoch}: loss={loss.item():.3f}")
    return model