# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Xm launch script for training with hyperparameter search."""

import itertools
from absl import app
from xmanager import xm
from xmanager import xm_local


def main(_) -> None:
  with xm_local.create_experiment(experiment_title='experiment') as experiment:
    spec = xm.PythonContainer(
        # Package the current directory that this script is in.
        path='../',
        base_image='gcr.io/deeplearning-platform-release/tf2-cpu',
        entrypoint=xm.ModuleName('preference.main'),
    )

    [executable] = experiment.package([
        xm.Packageable(
            executable_spec=spec,
            executor_spec=xm_local.Local.Spec(),
        ),
    ])

    lr = [1e-3]
    lambda_cosine_similarity = [0.2]  # [0.1, 0.2, 0.3, 0.4]
    lambda_correlation = [0.1]  # [0.1, 0.2, 0.3, 0.4]
    loss_fn = ['bradley_terry_bce']

    parameters = [
        {
            'data_dir': 'data/qa_preference',
            'cv_split': 1,
            'encoder_type': 'sentence-t5',
            'output_dim': 1,
            'dllp_loss_fn': lf,
            'aggregation_fn': 'avg',
            'loss_type': 'preference',
            'batch_size': 4,
            'lr': lr_hp,
            'epochs': 21,
            'last_layer_activation': 'sigmoid',
            'lambda_dllp': (
                1 - lambda_correlation_hp - lambda_cosine_similarity_hp
            ),
            'lambda_cosine_similarity': lambda_cosine_similarity_hp,
            'lambda_correlation': lambda_correlation_hp,
            'correlation': True,
            'cosine_similarity': True,
            'classify': True,
        }
        for lr_hp, lambda_cosine_similarity_hp, lambda_correlation_hp, lf in itertools.product(
            lr, lambda_cosine_similarity, lambda_correlation, loss_fn
        )
    ]

    for hparams in parameters:
      experiment.add(
          xm.Job(
              executable=executable,
              executor=xm_local.Local(),
              args=hparams,
          )
      )


if __name__ == '__main__':
  app.run(main)
