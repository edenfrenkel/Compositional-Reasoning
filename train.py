import argparse
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import json

from dataset import CompositionalDataset
from model import CompositionalRecognizer
from eval import eval_model


def main():
    parser = argparse.ArgumentParser(description='Train a compositional recognizer model.')
    parser.add_argument('-c', '--class-count', type=int, default=10)
    parser.add_argument('-l', '--seq-length', type=int, default=10)
    parser.add_argument('-o', '--overlap', type=int, default=2)
    parser.add_argument('--noise', type=float, default=None)
    parser.add_argument('-s', '--hidden-size', type=int, default=512)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-n', '--epoch-size', type=int, default=5120)
    parser.add_argument('-v', '--validation-size', type=int, default=5120)
    parser.add_argument('-e', '--epoch-count', type=int, default=100)
    parser.add_argument('-g', '--gpu_id', type=int, default=0)
    parser.add_argument('-d', '--dropout', type=float, default=0)
    parser.add_argument('-r', '--regenerate', action='store_true')
    parser.add_argument('-w', '--write-to', default=None)

    args = parser.parse_args()

    dataset = CompositionalDataset(args.class_count, args.seq_length, args.overlap, args.noise)
    model = CompositionalRecognizer(args.class_count, args.hidden_size, args.dropout)
    optimizer = torch.optim.Adam(model.parameters())
    device = torch.device('cuda:{id}'.format(id=args.gpu_id))

    if not args.regenerate:
        print('Generating training dataset...')
        dataloader = DataLoader(dataset.generate_dataset(args.epoch_size), args.batch_size, drop_last=True)
    print('Generating validation dataset...')
    val_dataloader = DataLoader(dataset.generate_dataset(args.validation_size), 256, drop_last=True)

    losses = []
    scores = []

    model = model.to(device)
    for epoch in range(1, args.epoch_count+1):
        model.train()
        total_loss = 0
        if args.regenerate:
            print('Generating training dataset for epoch {i}...'.format(i=epoch))
            dataloader = DataLoader(dataset.generate_dataset(args.epoch_size), args.batch_size, drop_last=True)
        print('Starting epoch {i}...'.format(i=epoch))
        pbar = tqdm(total=args.epoch_size, desc='Batch - (Loss = -)')
        batch = 1
        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            loss = model.forward_loss(x, labels)
            loss_val = loss.item()
            total_loss += loss_val

            loss.backward()
            optimizer.step()

            pbar.update(args.batch_size)
            pbar.set_description('Batch {b} (Loss = {ls})'.format(b=batch, ls=round(loss_val, 3)))
            batch += 1
        pbar.close()
        total_loss /= args.epoch_size // args.batch_size
        losses.append(total_loss)
        print('Average epoch loss:', total_loss)
        print('Evaluating MAP score...')
        model.eval()
        map_score = eval_model(model, val_dataloader, device)
        scores.append(map_score)
        print('Epoch MAP score:', map_score)

    if args.write_to is not None:
        with open(args.write_to, 'a') as f:
            f.write(json.dumps(dict(params=vars(args), losses=losses, scores=scores)) + '\n')


if __name__ == '__main__':
    main()
