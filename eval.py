import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score


def eval_model(model, dataloader, device):
    labels = []
    results = []
    for x, y in tqdm(dataloader):
        with torch.no_grad():
            x = x.to(device)
            res = torch.sigmoid(model(x)).detach()
            results.append(res.cpu().numpy())
            labels.append(y.numpy())

    map_score = average_precision_score(np.concatenate(labels), np.concatenate(results))

    return map_score
