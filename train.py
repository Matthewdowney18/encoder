import numpy as np
import torch
import torch.utils.data
import time

from dataset import SentenceDataset
from model import Seq2SeqModel, Seq2SeqModelAttention
from utils import variable, cuda, argmax, get_sentence_from_indices, \
    get_pretrained_embeddings, save_checkpoint, load_checkpoint, freeze_layer, \
    accuracy


def main():
    nb_epochs = 100
    batch_size = 500
    hidden_size = 256
    embedding_dim = 300
    pretrained_embeddings = None
    max_len = 20
    min_count = 2
    max_grad_norm = 5
    val_len = 10000
    weight_decay = 0.00001
    model_filename = '/home/mattd/pycharm/encoder/models3' \
                     '/Baseline'
    description_filename = \
        '/home/mattd/pycharm/encoder/description/description2.txt'
    output_file = '/home/mattd/pycharm/encoder/model_outputs_3/baseline'

    outfile = open(output_file, 'w')

    eng_fr_filename = '/home/okovaleva/projects/forced_apart/autoencoder/data' \
                      '/train_1M.txt'
    dataset = SentenceDataset(eng_fr_filename, max_len, min_count)
    string = 'Dataset: {}'.format(len(dataset))
    print(string)
    outfile.write(string+'\n')

    train_len = len(dataset) - val_len
    dataset_train, dataset_val = torch.utils.data.dataset.random_split(dataset, [train_len, val_len])
    string = 'Train {}, val: {}'.format(len(dataset_train), len(dataset_val))
    print(string)
    outfile.write(string+'\n')

    embeddings_dir = '/home/mattd/pycharm/encoder' \
                     '/embeddings_3min.npy'
    pretrained_embeddings = cuda(
        get_pretrained_embeddings(embeddings_dir, dataset))
    embedding_dim = pretrained_embeddings.shape[1]

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size, shuffle=False)

    vocab_size = len(dataset.vocab)
    padding_idx = dataset.vocab[SentenceDataset.PAD_TOKEN]
    init_idx = dataset.vocab[SentenceDataset.INIT_TOKEN]

    model = Seq2SeqModel(
        pretrained_embeddings, hidden_size, padding_idx, init_idx,
                         max_len, vocab_size, embedding_dim)
    model = cuda(model)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, amsgrad=True, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.vocab[SentenceDataset.PAD_TOKEN])

    model, optimizer, lowest_loss, description, last_epoch, \
    train_loss, val_loss = load_checkpoint(model_filename, model, optimizer)

    print(description)

    phases = ['train', 'val', ]
    data_loaders = [data_loader_train, data_loader_val, ]


    for epoch in range(last_epoch, last_epoch+nb_epochs):
        start = time.clock()

        #if epoch == 6:
        #    model.unfreeze_embeddings()
        #    parameters = list(model.parameters())
        #    optimizer = torch.optim.Adam(
        #        parameters, amsgrad=True, weight_decay=weight_decay)

        for phase, data_loader in zip(phases, data_loaders):
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = []
            epoch_sentenence_accuracy = []
            epoch_token_accuracy = []

            for i, inputs in enumerate(data_loader):
                optimizer.zero_grad()

                inputs = variable(inputs)
                targets = variable(inputs)

                outputs = model(inputs, targets)

                targets = targets.view(-1)
                outputs = outputs.view(targets.size(0), -1)

                loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                    optimizer.step()

                if phase == 'val':
                    predicted = torch.argmax(outputs.view(batch_size, max_len,
                                                          -1), -1)
                    batch_sentenence_accuracy, batch_token_accuracy = accuracy(
                        targets.view(batch_size, -1), predicted)
                    epoch_sentenence_accuracy.append(batch_sentenence_accuracy)
                    epoch_token_accuracy.append(batch_token_accuracy)
                epoch_loss.append(float(loss))

            epoch_loss = np.mean(epoch_loss)

            if phase == 'train':
                train_loss.append(epoch_loss)
                string = ('Epoch {:03d} | {} loss: {:.3f}'.format(
                    epoch, phase, epoch_loss))

                print(string, end='\n')
                outfile.write(string+'\n')
            else:
                averege_epoch_sentenence_accuracy = sum(epoch_sentenence_accuracy) / \
                    len(epoch_sentenence_accuracy)
                averege_epoch_token_accuracy = sum(epoch_token_accuracy) / \
                    len(epoch_token_accuracy)
                time_taken = time.clock() - start

                val_loss.append(epoch_loss)
                string = ' {} loss: {:.3f} | time: {:.3f}'.format(
                    phase, epoch_loss, time_taken)
                print(string, end='')

                string = '| sentence accuracy:{:.3f}| token accuracy:{:.3f}'.format(
                    averege_epoch_sentenence_accuracy, averege_epoch_token_accuracy)
                print(string, end='\n')
                outfile.write(string+'\n')
                if epoch_loss < lowest_loss:
                    save_checkpoint(
                        model, epoch_loss, optimizer, model_filename,
                        description_filename, epoch, train_loss, val_loss)
                    lowest_loss = epoch_loss



            # print random sentence
            if phase == 'val':
                random_idx = np.random.randint(len(dataset_val))
                inputs = dataset_val[random_idx]
                targets = inputs
                inputs_var = variable(inputs)

                outputs_var = model(inputs_var.unsqueeze(0)) # unsqueeze to get the batch dimension
                outputs = argmax(outputs_var).squeeze(0).data.cpu().numpy()

                string = '> {}'.format(get_sentence_from_indices(
                    inputs, dataset.vocab, SentenceDataset.EOS_TOKEN))
                print(string, end='\n')
                outfile.write(string+'\n')

                string = u'= {}'.format(get_sentence_from_indices(
                    targets, dataset.vocab, SentenceDataset.EOS_TOKEN))
                print(string, end='\n')
                outfile.write(string+'\n')

                string = u'< {}'.format(get_sentence_from_indices(
                    outputs, dataset.vocab, SentenceDataset.EOS_TOKEN))
                print(string, end='\n')
                outfile.write(string+'\n')
                print()
    outfile.close()


if __name__ == '__main__':
    main()