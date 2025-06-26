Setup:

- Download the dataset from [here](https://github.com/openai/prm800k/tree/main/prm800k/data) in fractal/data/prm800k/raw_data

Launch:

- Preprocess the dataset
  ```bash
  python math_reasoning/process_data.py 'data/prm800k/raw_data'
  ```
- Define the hyperparameters in xm_launch.py
- Run
  ```bash
  .venv/bin/xmanager launch math_reasoning/xm_launch.py
  ```

- To run experiments with hybrid prior, define the hyperparameters in hybrid_xm_launch.py
- Run
  ```bash
  .venv/bin/xmanager launch math_reasoning/hybrid_xm_launch.py
  ```
