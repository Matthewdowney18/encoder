import torch
from matplotlib import pyplot

from utils import load_checkpoint
from model import Seq2SeqModel, Seq2SeqModelAttention
from dataset import SentenceDataset

def plot_data(sample_1, sample_2):
    pyplot.figure(1)
    pyplot.plot(sample_1, 'r-', sample_2, 'b-')
    pyplot.title('i dont know man')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.show()


def main():
    hidden_size = 256
    embedding_dim = 300
    pretrained_embeddings = None
    max_len = 20
    min_count = 2
    max_grad_norm = 5
    val_len = 10000
    weight_decay = 0.00001


    model_filename_1 = '/home/mattd/pycharm/encoder/models3/Baseline'
    model_filename_2 = '/home/mattd/pycharm/encoder/models3/Attention'

    eng_fr_filename = '/home/okovaleva/projects/forced_apart/autoencoder/data' \
                      '/train_1M.txt'
    dataset = SentenceDataset(eng_fr_filename, max_len, min_count)

    vocab_size = len(dataset.vocab)
    padding_idx = dataset.vocab[SentenceDataset.PAD_TOKEN]
    init_idx = dataset.vocab[SentenceDataset.INIT_TOKEN]


    model = Seq2SeqModel(
        pretrained_embeddings, hidden_size, padding_idx, init_idx,
        max_len, vocab_size, embedding_dim)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, amsgrad=True,
                                 weight_decay=weight_decay)

    model, optimizer, lowest_loss, description, last_epoch, \
    train_loss_1, val_loss_1 = load_checkpoint(model_filename_1, model,
                                               optimizer)


    model = Seq2SeqModelAttention(
        pretrained_embeddings, hidden_size, padding_idx, init_idx,
        max_len, vocab_size, embedding_dim)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, amsgrad=True,
                                 weight_decay=weight_decay)

    model, optimizer, lowest_loss, description, last_epoch, \
    train_loss_2, val_loss_2 = load_checkpoint(model_filename_2, model,
                                              optimizer)

    plot_data(train_loss_1, val_loss_1)
    plot_data(train_loss_2, val_loss_2)

if __name__ == '__main__':
    main()