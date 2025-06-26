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

"""XManager launch script for hybrid prior experiments on MultiSpanQA."""
import itertools

from absl import app
from xmanager import xm
from xmanager import xm_local


dataset = 'MultiSpanQA'


def main(_) -> None:
  with xm_local.create_experiment(experiment_title='experiment') as experiment:
    spec = xm.PythonContainer(
        # Package the current directory that this script is in.
        path='../../',
        base_image='gcr.io/deeplearning-platform-release/tf2-cpu',
        entrypoint=xm.ModuleName('retrieval.MultiSpanQA.hybrid_prior_main'),
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
    # lr = [1e-3, 1e-4, 1e-5]
    # lambda_dllp = [1, 0.8, 0.6, 0.5, 0.2, 0]
    # cosine_similarity = [True, False]
    # loss_type = ['aggregate', 'instance']
    lr = [1e-3]
    loss_fn = ['bce']
    # loss_type = ['aggregate']
    # lambda_cosine_similarity = [0.1, 0.2, 0.3, 0.4, 0.5]
    lambda_cosine_similarity = [0.1]
    # lambda_correlation = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    lambda_correlation = [0.1]
    # lambda_hybrid_prior = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    lambda_hybrid_prior = [0.1]

    parameters = [
        {
            'encoder_type': 'sentence-t5',
            'output_dim': 1,
            'dllp_loss_fn': lf,
            'aggregation_fn': 'max',
            'loss_type': 'aggregate',
            'batch_size': 80,
            'lr': lr_hp,
            'epochs': 1,
            'last_layer_activation': 'sigmoid',
            'lambda_dllp': (
                1
                - lambda_correlation_hp
                - lambda_cosine_similarity_hp
                - lambda_hybrid_prior_hp
            ),
            'lambda_cosine_similarity': lambda_cosine_similarity_hp,
            'lambda_correlation': lambda_correlation_hp,
            'lambda_hybrid_prior': lambda_hybrid_prior_hp,
            'correlation': True,
            'cosine_similarity': True,
            'hybrid': True,
            'classify': True,
            'use_curriculum': False,
            'update_after_epochs': 5,
        }
        for lr_hp, lambda_cosine_similarity_hp, lambda_correlation_hp, lambda_hybrid_prior_hp, lf in itertools.product(
            lr,
            lambda_cosine_similarity,
            lambda_correlation,
            lambda_hybrid_prior,
            loss_fn,
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
