Setup:

- Download the dataset from [here](https://github.com/haonan-li/MultiSpanQA/tree/master/data/MultiSpanQA_data) in fractal/data/multispan_qa

Launch:

- Preprocess the dataset
  ```bash
  python retrieval/MultiSpanQA/process_data.py 'data/multispan_qa'
  ```
- Define the hyperparameters in xm_launch.py
- Run
  ```bash
  .venv/bin/xmanager launch retrieval/MultiSpanQA/xm_launch.py
  ```

- To run experiments with hybrid prior, define the hyperparameters in hybrid_xm_launch.py
- Run
  ```bash
  .venv/bin/xmanager launch retrieval/MultiSpanQA/hybrid_xm_launch.py
  ```
