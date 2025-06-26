Setup:

- Download the WikiCatSum dataset from [here](https://datashare.ed.ac.uk/handle/10283/3368) in fractal/data/preprocessed_wikicatsum/
- Download the AquaMuSe dataset from [here](https://github.com/google-research-datasets/aquamuse/tree/main/v3) in fractal/data/preprocessed_aquamuse/

Launch:

- Preprocess the dataset

  Execute all the cells in fake_summary.ipynb to generate fake summaries and subsequently create a TensorFlow dataset. The fake summaries were generated using PaLM 2 Bison Chat. Instructions for setting up and using this model can be found [here](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/chat-bison?inv=1&invt=AbytLg).

- Define the hyperparameters in xm_launch.py
- Run
  ```bash
  .venv/bin/xmanager launch entailment/xm_launch.py
  ```

- To run experiments with hybrid prior, define the hyperparameters in hybrid_xm_launch.py
- Run
  ```bash
  .venv/bin/xmanager launch entailment/hybrid_xm_launch.
  ```
