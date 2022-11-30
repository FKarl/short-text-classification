## NICE
### The files used to create this dataset are downloaded from the [World Intellectual Property Organization (WIPO)](https://www.wipo.int/nice/its4nice/ITSupport_and_download_area/20220101/MasterFiles/index.html).
<sup>Retrieved July 17, 2022</sup>

>  ncl-20220101-classification_top_structure-20210623.zip >
   ncl-20220101-classification_top_structure-20210623.xml as [labels.xml](source/labels.xml)

> ncl-20220101-classification_texts-20210623.zip > 
  ncl-20220101-en-classification_texts-20210623.xml as [texts.xml](source/labels.xml)

### Setup

To create the dataset, put the downloaded files in the following structure:

    .
    ├── NICE
    ├── NICE_binary
    ├── creation.ipynb
    ├── creation.py
    ├── README.md
    └── source
        ├── labels.xml
        └── texts.xml

Now run the [`creation.py`](creation.py) script or [`creation.ipynb`](creation.ipynb) notebook. This will create the `NICE-45` and `NICE-2` datasets.
