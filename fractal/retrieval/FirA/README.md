Setup:

- Download the dataset from [here](https://github.com/sebastian-hofstaetter/fira-trec-19-dataset/tree/master/data) in fractal/data/fira
- From the above dataset link save:
  - `data/input/documents.tsv` as data/fira/documents.tsv
  - `data/input/queries.tsv` as data/fira/queries.tsv
  - `data/fira-trec19-raw-judgements.tsv` as data/fira/fira-trec19-raw-judgements.tsv

Launch:

- Preprocess the dataset
  ```bash
  python retrieval/FirA/process_data.py 'data/fira'
  ```
- Define the hyperparameters in xm_launch.py
- Run
  ```bash
  .venv/bin/xmanager launch retrieval/FirA/xm_launch.py
  ```
