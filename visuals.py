import json
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Visualize training results.')
    parser.add_argument('-f', '--filename', default=None)
    parser.add_argument('-e', '--epochs', type=int, default=None)
    parser.add_argument('-s', '--hidden-size', action='store_true')
    parser.add_argument('-n', '--noise', action='store_true')

    args = parser.parse_args()
    if args.filename is None:
        return

    fig, (loss_ax, score_ax) = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    loss_ax.set_ylabel('Train Loss')
    loss_ax.set_xlabel('Epoch')
    score_ax.set_ylabel('mAP')
    score_ax.set_xlabel('Epoch')
    score_ax.set_ylim([0.6, 1])

    with open(args.filename, 'r') as f:
        line = f.readline()
        while line:
            data = json.loads(line)
            name = '{c} classes'.format(c=data['params']['class_count'])
            if args.hidden_size:
                name += ', hidden size={n}'.format(n=data['params']['hidden_size'])
            if args.noise:
                name += ', noise={n}'.format(n=data['params']['noise'])
            epochs = data['params']['epoch_count'] if args.epochs is None else args.epochs
            epoch_list = list(range(epochs))
            loss_ax.plot(epoch_list, data['losses'][:epochs], label=name)
            score_ax.plot(epoch_list, data['scores'][:epochs], label=name)
            line = f.readline()

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
