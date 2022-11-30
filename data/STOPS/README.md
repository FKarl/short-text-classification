## Short Texts Of Products and Services (STOPS)
### The files used to create this dataset are downloaded from:

>  [MAVE](https://github.com/google-research-datasets/MAVE)

<sup>Retrieved July 17, 2022</sup>

>  yelp_dataset > [yelp_academic_dataset_business.json](https://www.yelp.com/dataset/download)

<sup>Retrieved July 17, 2022</sup>

### Setup

To create the dataset, put the downloaded files in the following structure:

    .
    ├── STOPS
    ├── STOPS-2
    ├── creation.ipynb
    ├── creation.py
    ├── README.md
    └── source
        ├── mave
        │     ├── mave_negatives.jsonl
        │     └── mave_positives.jsonl
        └── yelp
            └── yelp_academic_dataset_business.json

Now run the [`creation.py`](creation.py) script or [`creation.ipynb`](creation.ipynb) notebook. This will create the `STOPS-41` and `STOPS-2` datasets.
