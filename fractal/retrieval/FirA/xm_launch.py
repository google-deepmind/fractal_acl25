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

"""XManager launch file for FirA retrieval task."""

import itertools

from absl import app
from xmanager import xm
from xmanager import xm_local


def main(_) -> None:
  with xm_local.create_experiment(experiment_title='experiment') as experiment:
    spec = xm.PythonContainer(
        # Package the current directory that this script is in.
        path='../../',
        base_image='gcr.io/deeplearning-platform-release/tf2-cpu',
        entrypoint=xm.ModuleName('retrieval.FirA.main'),
    )

    [executable] = experiment.package([
        xm.Packageable(
            executable_spec=spec,
            executor_spec=xm_local.Local.Spec(),
        ),
    ])

    # Hyper-parameter definition.
    # Note that the hyperparameter arguments, if used, must correspond to flags
    # defined in your training binary.
    lr = [1e-3]  # [1e-3, 1e-4, 1e-5]
    lambda_dllp = [0.8]
    cosine_similarity = [False]
    lambda_correlation = [0.1]
    loss_fn = ['ce']

    parameters = [
        {
            'encoder_type': 'sentence-t5',
            'output_dim': 4,
            'dllp_loss_fn': lf,
            'aggregation_fn': 'max',
            'loss_type': 'aggregate',
            'batch_size': 2,
            'lr': lr_hp,
            'epochs': 1,
            'last_layer_activation': 'softmax',
            'lambda_dllp': lambda_dllp_hp,
            'lambda_cosine_similarity': (
                1 - lambda_dllp_hp - lambda_correlation_hp
            ),
            'lambda_correlation': lambda_correlation_hp,
            'correlation': False,
            'cosine_similarity': cos_sim,
            'classify': False,
            'cv_split': 1,
        }
        for lr_hp, lambda_dllp_hp, cos_sim, lf, lambda_correlation_hp in itertools.product(
            lr, lambda_dllp, cosine_similarity, loss_fn, lambda_correlation
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
