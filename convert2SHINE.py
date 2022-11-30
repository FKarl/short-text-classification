import json

from data import load_data


def convert_data(name):
    dataset = load_data(name)
    train_texts, train_labels = dataset['train']
    test_texts, test_labels = dataset['test']
    label_dict = dataset['label_dict']

    path_to_save = './SHINE/preprocess/'
    main_dict = dict()
    train_dict = dict()
    test_dict = dict()

    for i, text in enumerate(train_texts):
        element = dict()
        element['text'] = text
        element['label'] = label_dict[train_labels[i]]
        train_dict[i] = element

    for i, text in enumerate(test_texts):
        element = dict()
        element['text'] = text
        element['label'] = label_dict[test_labels[i]]
        test_dict[i] = element

    main_dict['train'] = train_dict
    main_dict['test'] = test_dict

    # drop json to file
    with open(path_to_save + name + '_split.json', 'w') as f:
        json.dump(main_dict, f)


if __name__ == '__main__':
    datasets = ['MR', 'TREC', 'SST2', 'R8', 'Twitter', 'SearchSnippets', 'NICE', 'NICE2', 'STOPS', 'STOPS2']
    for dataset in datasets:
        convert_data(dataset)
        print(f'convert {dataset} done.')
