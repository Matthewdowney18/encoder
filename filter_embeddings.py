from vecto import embeddings
import numpy as np
import pickle

from dataset import SentenceDataset


def get_vectors(filename):
    objects = dict()
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                objects = pickle.load(openfile)
            except EOFError:
                break
    return objects

def main():
    embeddings_dir = '/home/okovaleva/projects/forced_apart/autoencoder/data' \
                      '/w2vec.pkl'
    eng_fr_filename = '/home/okovaleva/projects/forced_apart/autoencoder/data' \
                      '/train.txt'
    eng_fr_filename2 = '/home/okovaleva/projects/forced_apart/autoencoder/data' \
                      '/test.txt'

    dataset = SentenceDataset(eng_fr_filename, 20, 2)
    #dataset.add_file(eng_fr_filename2)
    dataset.vocab.prune_vocab(2)

    vectors = get_vectors(embeddings_dir)

    #emb = embeddings.load_from_dir(embeddings_dir)

    embs_matrix = np.zeros((len(dataset.vocab), vectors['r'].size))

    for i, token in enumerate(dataset.vocab.token2id):
        if token in vectors:
            embs_matrix[i] = vectors[token]
    np.save('embeddings_2min', embs_matrix)


if __name__ == '__main__':
    main()