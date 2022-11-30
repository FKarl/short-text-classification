from nltk.stem.porter import *
from tqdm import tqdm

from data import load_data


def convert_data(name):
    dataset = load_data(name)
    train_texts, train_labels = dataset['train']
    test_texts, test_labels = dataset['test']
    label_dict = dataset['label_dict']

    path_to_data = 'DADGNN/content/data'

    # label.txt
    with open(f'{path_to_data}/{name}/label.txt', 'w', encoding="utf-8") as f:
        for label in list(label_dict.keys())[:-1]:
            f.write(str(label) + '\n')
        f.write(str(list(label_dict.keys())[-1]))

    # remove newlines from texts
    train_texts = [text.replace('\n', ' ') for text in train_texts]
    test_texts = [text.replace('\n', ' ') for text in test_texts]
    train_texts = [text.replace('\r', ' ') for text in train_texts]
    test_texts = [text.replace('\r', ' ') for text in test_texts]
    train_texts = [text.replace('\n\r', ' ') for text in train_texts]
    test_texts = [text.replace('\n\r', ' ') for text in test_texts]

    # train.txt
    with open(f'{path_to_data}/{name}/{name}-train.txt', 'w', encoding="utf-8") as f:
        for text, label in zip(train_texts[:-1], train_labels[:-1]):
            f.write(f'{label}\t{text}\n')
        f.write(f'{train_labels[-1]}\t{train_texts[-1]}')

    # test.txt
    with open(f'{path_to_data}/{name}/{name}-test.txt', 'w', encoding="utf-8") as f:
        for text, label in zip(test_texts[:-1], test_labels[:-1]):
            f.write(f'{label}\t{text}\n')
        f.write(f'{test_labels[-1]}\t{test_texts[-1]}')

    # region vocab.txt
    print('building vocab')

    words = []
    for text in train_texts + test_texts:
        words += text.split()

    # stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    print(words[:10])

    print()
    # filter out words that occur less than 5 times
    counter = dict()
    for word in tqdm(words):
        if word in counter:
            counter[word] += 1
        else:
            counter[word] = 1

    words = [word for word in words if counter[word] >= 5]
    words = list(set(words))

    print(f'{len(words)} words found')

    with open(f'{path_to_data}/{name}/{name}-vocab.txt', 'w', encoding="utf-8") as f:
        # write UNK token
        f.write('UNK\n')
        for word in words[:-1]:
            f.write(word + '\n')
        f.write(words[-1])
    # endregion


if __name__ == '__main__':
    datasets = ['MR', 'TREC', 'SST2', 'R8', 'Twitter', 'SearchSnippets', 'NICE', 'NICE2', 'STOPS', 'STOPS2']
    for dataset in datasets:
        convert_data(dataset)
        print(f'convert {dataset} done.')
