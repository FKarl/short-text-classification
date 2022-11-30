# Transformers are Short Text Classifiers: A Study of Inductive Short Text Classifiers on Benchmarks and Real-world Datasets

This repository contains code to reproduce the results in our paper "Transformers are Short Text Classifiers: A Study of Inductive Short Text Classifiers on Benchmarks and Real-world Datasets".

This study's objective was to examine the performance of a variety of short text classifiers as well as the top performing traditional text classifier on single-label short text classification.
Furthermore, we propose in this work two new real-world datasets for short text classification (e.g. [STOPS](./data/STOPS) and [NICE](./data/NICE)).

## Table of Contents
- [Getting Started](#getting-started)
- [Running the Experiments](#running-the-experiments)
- [Structure of the Repository](#structure-of-the-repository)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Getting Started

These instructions will let you run the code on your local machine for reproduction purposes.


### Installing

A step by step series that tell you how to get the experiments running.

Install the requirements using pip

```bash
pip install -r requirements.txt
```

Make sure you installed the right CUDA version for your GPU. 
You can check the CUDA version of your GPU [here](https://developer.nvidia.com/cuda-gpus).

### Setup the Datasets

Not all datasets are included in this repository due to licensing issues.
To run the experiments, you need to download the datasets and place them in the correct folder.

For instructions on how to obtain the data, see the [README](data/README.md) in the data folder.

## Running the Experiments

To run the experiments, you can use the following command:

```bash
python main.py <dataset> <model>
```

where `<dataset>` is the name of the dataset and `<model>` is the name of the model.
Possible entries for `<dataset>` are: 

-  `MR`
-  `R8`
-  `SearchSnippets`
-  `Twitter`
-  `TREC`
-  `SST2`
-  `NICE`
-  `NICE2`
-  `STOPS`
-  `STOPS2`


Possible entries for `<model>` are:

- `BERT`
- `ROBERTA`
- `DEBERTA`
- `MLP`
- `ERNIE`
- `DISTILBERT`
- `ALBERT`
- `LSTM`
- `STACKING`
- `WEIGHTED_BOOST`
- `WEIGHTED`

`STACKING`, `WEIGHTED_BOOST` and `WEIGHTED` are ensemble methods that require additional parameters.
For `WEIGHTED_BOOST` and `WEIGHTED` you can specify the models that should be used in the ensemble by adding the parameters `--m1` and `--m2` followed by the model names.
For `STACKING` you also need to specify a meta model by adding the parameter `--mm` followed by the meta model name.

For information on optional parameters, you can use the `--help` flag.

```bash
python main.py --help
```

### Sample 

To run the experiments on the MR dataset using the ALBERT model and our parameters, you can use the following command:

```bash
python main.py MR ALBERT --learning_rate=1e-5 --batch_size=32 --num_train_epochs=10 --dropout=0
```

### Scripts
There are also scripts to run the experiments on all datasets with our selected parameters.
These scripts can be found in the [run_scripts](run_scripts) folder.

For further information on the scripts, see the [README](run_scripts/README.md) in the run_scripts folder.

## Structure of the repository
The repository is structured as follows:

    .
    ├── data                    # Data files
    ├── run_script              # Bash scripts to run all experiments
    ├── convert2dadgnn.py       # Script to convert data to DADGNN format
    ├── convert2inductTGCN.py   # Script to convert data to InductTGCN format
    ├── convert2SHINE.py        # Script to convert data to SHINE format
    ├── data.py                 # Data loading and representation
    ├── ensemble_models.py      # Ensemble models declaration
    ├── models.py               # MLP and LSTM declaration
    ├── main.py                 # Main script to run the experiments
    └── requirements.txt        # Requirements file

The source code of the foreign models is not included in this repository.
You can find the source code of the foreign models in the following repositories:

- [InductTGCN](https://github.com/usydnlp/InductTGCN)
- [SHINE](https://github.com/tata1661/SHINE-EMNLP21)
- DADGNN: Not publicly available

## License

This project is licensed under the [MIT](LICENSE) License - see the [LICENSE](LICENSE) file for details


## Acknowledgments

We would like to thank the authors of the following repositories for making their code publicly available:
- The WideMLP code was adopted from [Lukas Galke](https://github.com/lgalke/text-clf-baselines)
