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

"""xm_launch file for entailement task."""
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
        entrypoint=xm.ModuleName('entailment.entailment'),
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
    lr = [1e-4]
    lambda_dllp = [0.8]  # [1, 0.8, 0.6, 0.5, 0.4, 0.2]
    # lambda_dllp = [0.5]
    lambda_correlation = [0.1]  # [0.5, 0.4, 0.2,0]
    # lambda_correlation = [0.5]
    # cosine_similarity = [True, False]
    loss_fn = ['bce']
    frac_train_1bags = [0.5]  # [0.3,0.4,0.5,0.6,0.7]
    # frac_train_1bags = [0.5]

    parameters = [
        {
            'encoder_type': 'sentence-t5',
            'output_dim': 1,
            'loss_fn': lf,
            'loss_type': 'aggregate',
            'batch_size': 256,
            'lr': lr_hp,
            'epochs': 1,
            'last_layer_activation': 'sigmoid',
            'lambda_dllp': lambda_dllp_hp,
            'lambda_cosine_similarity': (
                1 - lambda_dllp_hp - lambda_correlation_hp
            ),
            'lambda_correlation': lambda_correlation_hp,
            'correlation': True,
            'cosine_similarity': True,
            'classify': True,
            'frac_train_1bags': frac_1bags,
        }
        for lr_hp, lambda_dllp_hp, lambda_correlation_hp, lf, frac_1bags in itertools.product(
            lr, lambda_dllp, lambda_correlation, loss_fn, frac_train_1bags
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
