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

"""Main function for the math reasoning model with hybrid prior."""

import datetime
import json
import os

from absl import app
from absl import flags
from math_reasoning.hybrid_model import get_pseudolabels_loader
from math_reasoning.hybrid_model import SentenceT5_MLP_Agg_Hyb_Prior
from math_reasoning.utils import get_dataloader_hybrid_prior
import tensorflow as tf


flags.DEFINE_string('encoder_type', default='sentence-t5', help='Encoder type')
flags.DEFINE_integer('output_dim', default=1, help='Output dimension')
flags.DEFINE_string('dllp_loss_fn', default='mae', help='Loss function')
flags.DEFINE_string(
    'loss_type', default='aggregate', help='Aggregate or instance loss'
)
flags.DEFINE_integer('batch_size', default=2048, help='Batch size')
flags.DEFINE_float('lr', default=1e-5, help='Learning rate')
flags.DEFINE_integer('epochs', default=40, help='Number of epochs')
flags.DEFINE_string(
    'last_layer_activation', default='sigmoid', help='last layer activation'
)
flags.DEFINE_float('lambda_dllp', default=1.0, help='lambda bag loss prior')
flags.DEFINE_float(
    'lambda_cosine_similarity',
    default=0.0,
    help='lambda cosine similarity prior',
)
flags.DEFINE_float(
    'lambda_correlation', default=0.0, help='lambda correlation prior'
)
flags.DEFINE_bool('correlation', default=False, help='Use correlation prior')
flags.DEFINE_bool(
    'cosine_similarity', default=False, help='Use Cosine similarity prior'
)
flags.DEFINE_bool(
    'classify', default=True, help='classification task True/False'
)
flags.DEFINE_bool('hybrid', default=True, help='Use hybrid prior')
flags.DEFINE_float(
    'lambda_hybrid_prior', default=0.0, help='lambda hybrid prior'
)
flags.DEFINE_float(
    'frac_disagg_bags',
    default=0.2,
    help='Fraction of bags to disaggregate (instances)',
)
flags.DEFINE_string(
    'aggregation_fn', default='min', help='Aggregation function'
)
flags.DEFINE_string('tpu', 'local', 'The BNS address of the first TPU worker.')
flags.DEFINE_integer(
    'cv_split', default=1, help='Cross-validation split'
)
flags.DEFINE_string('data_dir', default='data/prm800k/raw_data', help='Data directory')

FLAGS = flags.FLAGS


def main(_) -> None:
  data_dir = FLAGS.data_dir

  instance_model = SentenceT5_MLP_Agg_Hyb_Prior(
      output_dim=FLAGS.output_dim,
      last_layer_activation=FLAGS.last_layer_activation,
      cosine_similarity=False,
      correlation=False,
      hybrid_prior=False,
      lambda_dllp=1.0,
      lambda_cosine_similarity=0.0,
      lambda_correlation=0.0,
      lambda_hybrid_prior=0.0,
      lr=FLAGS.lr,
      classify=True,
      loss_type='instance',
      dllp_loss_fn='bce',
      aggregation_fn='min',
      instance_model=None
  )

  eid = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  config_str = json.dumps(FLAGS.flag_values_dict(), indent=2)  # pylint: disable=unused-variable

  os.makedirs(f'{data_dir}/logs/{eid}', exist_ok=True)
  os.makedirs(f'{data_dir}/checkpoint/{eid}', exist_ok=True)
  with open(f'{data_dir}/logs/{eid}/config.json', 'w') as f:
    f.write(config_str)
  with open(f'{data_dir}/checkpoint/{eid}/config.json', 'w') as f:
    f.write(config_str)
  ckpt_folder = f'{data_dir}/checkpoint/{eid}'

  checkpoint_filepath = ckpt_folder + '/epoch-{epoch:02d}/ckpt.weights.h5'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(  # pylint: disable=unused-variable
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_test_loss',
      mode='min',
      save_best_only=True,
  )

  train_instances_loader, trainloader, valloader, testloader = (
      get_dataloader_hybrid_prior(
          data_dir,
          batch_size=FLAGS.batch_size,
          frac_disagg_bags=FLAGS.frac_disagg_bags,
          cv_split=FLAGS.cv_split,
      )
  )

  instance_model.compile(run_eagerly=True, steps_per_execution=1)

  instance_model.fit(
      train_instances_loader,
      epochs=FLAGS.epochs,
      validation_data=valloader,
      # steps_per_epoch=20
  )

  if FLAGS.loss_type == 'aggregate':

    model = SentenceT5_MLP_Agg_Hyb_Prior(
        output_dim=FLAGS.output_dim,
        last_layer_activation=FLAGS.last_layer_activation,
        cosine_similarity=FLAGS.cosine_similarity,
        correlation=FLAGS.correlation,
        hybrid_prior=FLAGS.hybrid,
        lambda_dllp=FLAGS.lambda_dllp,
        lambda_cosine_similarity=FLAGS.lambda_cosine_similarity,
        lambda_correlation=FLAGS.lambda_correlation,
        lambda_hybrid_prior=FLAGS.lambda_hybrid_prior,
        lr=FLAGS.lr,
        classify=FLAGS.classify,
        loss_type=FLAGS.loss_type,
        dllp_loss_fn=FLAGS.dllp_loss_fn,
        aggregation_fn=FLAGS.aggregation_fn,
        instance_model=instance_model,
    )

    model.compile(run_eagerly=True, steps_per_execution=1)

    os.makedirs(f'{data_dir}/logs/{eid}', exist_ok=True)
    os.makedirs(f'{data_dir}/checkpoint/{eid}_prior', exist_ok=True)
    config_str = FLAGS.flags_into_string()
    with open(f'{data_dir}/logs/{eid}/config.json', 'w') as f:
      f.write(config_str)
    with open(f'{data_dir}/checkpoint/{eid}_prior/config.json', 'w') as f:
      f.write(config_str)

    model.fit(
        trainloader,
        epochs=FLAGS.epochs,
        validation_data=valloader,
        # steps_per_epoch=20
    )

    model.evaluate(valloader)

    pslab_loader = get_pseudolabels_loader(
        model, trainloader, FLAGS.aggregation_fn, FLAGS.batch_size
    )
    psl_model = SentenceT5_MLP_Agg_Hyb_Prior(
        output_dim=FLAGS.output_dim,
        last_layer_activation=FLAGS.last_layer_activation,
        cosine_similarity=False,
        correlation=False,
        hybrid_prior=False,
        lambda_dllp=1.0,
        lambda_cosine_similarity=0.0,
        lambda_correlation=0.0,
        lambda_hybrid_prior=0.0,
        lr=FLAGS.lr,
        classify=True,
        loss_type='instance',
        dllp_loss_fn='bce',
        aggregation_fn='min',
        instance_model=instance_model,
    )
    psl_model.compile(run_eagerly=True, steps_per_execution=1)
    psl_model.fit(
        pslab_loader,
        epochs=FLAGS.epochs,
        validation_data=valloader,
        # steps_per_epoch=20
    )
    psl_model.evaluate(
        testloader
    )
if __name__ == '__main__':
  app.run(main)
