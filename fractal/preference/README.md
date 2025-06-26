Setup:

- Download the dataset from [here](https://github.com/allenai/FineGrainedRLHF/tree/main/tasks/qa_feedback/data) in fractal/data/qa_preference

Launch:

- Preprocess the dataset
  ```bash
  python preference/process_data.py 'data/qa_preference'
  ```
- Define the hyperparameters in xm_launch.py
- Run
  ```bash
  .venv/bin/xmanager launch preference/xm_launch.py
  ```
