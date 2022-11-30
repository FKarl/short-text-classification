from tqdm import tqdm

from data import load_data


def convert_data(name):
    dataset = load_data(name)
    train_texts, train_labels = dataset['train']
    test_texts, test_labels = dataset['test']
    label_dict = dataset['label_dict']

    path_to_data = 'HYPERGAT/data'

    # labels.txt
    with open(f'{path_to_data}/{name}_labels.txt', 'w', encoding="utf-8") as f:
        counter = 0
        for l in train_labels:
            f.write(f'{counter}\ttrain\t{l}\n')
            counter += 1
        for l in test_labels:
            f.write(f'{counter}\ttest\t{l}\n')
            counter += 1

    # corpus.txt
    with open(f'{path_to_data}/{name}_corpus.txt', 'w', encoding="utf-8") as f:
        for text in tqdm(train_texts):
            f.write(f'{text}\n')
        for text in tqdm(test_texts):
            f.write(f'{text}\n')


if __name__ == '__main__':
    datasets = ['MR', 'TREC', 'SST2', 'R8', 'Twitter', 'SearchSnippets', 'NICE', 'NICE2', 'STOPS', 'STOPS2']
    for dataset in datasets:
        convert_data(dataset)
        print(f'convert {dataset} done.')
