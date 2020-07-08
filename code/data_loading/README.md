## A Python data adapter for the extractor model data

#####Adapter.py
This script adapts the wikihow dataset original `.csv` file into train/test/val splits in the form of `.json` files,
 each containing `json` objects corresponding to the articles/summaries pairs in the dataset.
This is needed to have the correctly formatted input for the extractor **SummaRuNNer** model. The dataset csv file should be downloaded independently. 
An intermediate tokenized `.csv` file will be produced before dumping the entries into the `json`s objects for a matter of efficiency. 

If you wish to modify the tokenization you need to manually delete the intermediate file, otherwise any changes will be
 disregarded and the old tokenization will still be used for the dumping. 

### Setup -- Needed libraries

```
numpy
pandas
nltk
argspare
json
sklearn
```

### Usage  

```shell
python adapter.py -csv_to_json XXX.csv 
```

Will output a `.csv` file and a `train/test/val.json` files in the same directory of the input file.
