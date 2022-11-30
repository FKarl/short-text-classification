from data import load_data

def convert_data(name):
    dataset = load_data(name)
    train_texts, train_labels = dataset['train']
    test_texts, test_labels = dataset['test']
    label_dict = dataset['label_dict']

    path_to_data = 'InductTGCN/data'

    # dataset.csv
    with open(f'{path_to_data}/{name}.csv', 'w', encoding="utf-8") as f:
        f.write(f'text,label,train\n')
        for text, label in zip(train_texts, train_labels):
            text = text.replace('"', '""')
            f.write(f'"{text}",{label},train\n')
        for text, label in zip(test_texts, test_labels):
            text = text.replace('"', '""')
            f.write(f'"{text}",{label},test\n')


if __name__ == '__main__':
    datasets = ['MR', 'TREC', 'SST2', 'R8', 'Twitter', 'SearchSnippets', 'NICE', 'NICE2', 'STOPS', 'STOPS2']
    for dataset in datasets:
        convert_data(dataset)
        print(f'convert {dataset} done.')
