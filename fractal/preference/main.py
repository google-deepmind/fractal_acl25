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

"""Main file."""

import datetime
import json
import os

from absl import app
from absl import flags
from preference.model import SentenceT5_MLP_Preference
from preference.utils import CustomTBCallback
from preference.utils import get_dataloader
import tensorflow as tf


flags.DEFINE_string(
    'data_dir',
    default='data/qa_preference',
    help='raw data location',
)
flags.DEFINE_integer('cv_split', default=1, help='Cross validation split')
flags.DEFINE_bool(
    'classify', default=False, help='Classification task True/False'
)
flags.DEFINE_bool(
    'correlation', default=False, help='Correlation prior True/False'
)
flags.DEFINE_bool(
    'cosine_similarity',
    default=False,
    help='Cosine similarity prior True/False',
)
flags.DEFINE_float(
    'lambda_correlation', default=0.0, help='Correlation prior lambda'
)
flags.DEFINE_float(
    'lambda_cosine_similarity',
    default=0.0,
    help='Cosine similarity prior lambda',
)
flags.DEFINE_float('lambda_dllp', default=1.0, help='Bag loss lambda')
flags.DEFINE_float('lr', default=1e-5, help='Learning rate')
flags.DEFINE_integer('batch_size', default=2048, help='Batch size')
flags.DEFINE_integer('epochs', default=200, help='Number of epochs')
flags.DEFINE_integer('output_dim', default=1, help='Output dimension')
flags.DEFINE_string(
    'aggregation_fn', default='max', help='Aggregation function'
)
flags.DEFINE_string('dllp_loss_fn', default='mae', help='Loss function')
flags.DEFINE_string('encoder_type', default='sentence-t5', help='Encoder type')
flags.DEFINE_string(
    'last_layer_activation', default='sigmoid', help='Last layer activation'
)
flags.DEFINE_string(
    'loss_type', default='aggregate', help='Loss type aggregate/instance'
)

FLAGS = flags.FLAGS


def main(argv) -> None:  # pylint: disable=unused-argument

  model = SentenceT5_MLP_Preference(
      output_dim=FLAGS.output_dim,
      last_layer_activation=FLAGS.last_layer_activation,
      cosine_similarity=FLAGS.cosine_similarity,
      correlation=FLAGS.correlation,
      lambda_dllp=FLAGS.lambda_dllp,
      lambda_cosine_similarity=FLAGS.lambda_cosine_similarity,
      lambda_correlation=FLAGS.lambda_correlation,
      lr=FLAGS.lr,
      classify=FLAGS.classify,
      loss_type=FLAGS.loss_type,
      dllp_loss_fn=FLAGS.dllp_loss_fn,
      aggregation_fn=FLAGS.aggregation_fn,
      data_dir=FLAGS.data_dir,
  )

  data_dir = FLAGS.data_dir
  trainloader, valloader = get_dataloader(
      FLAGS.data_dir, batch_size=FLAGS.batch_size, cv_split=FLAGS.cv_split
  )
  model.compile(run_eagerly=True, steps_per_execution=1)

  eid = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  config_str = json.dumps(FLAGS.flag_values_dict(), indent=2)
  tb_callback = CustomTBCallback(
      hyperparams=config_str,
      log_dir=f'{data_dir}/logs/{eid}/',
      update_freq='epoch',
  )

  os.makedirs(f'{data_dir}/logs/{eid}', exist_ok=True)
  os.makedirs(f'{data_dir}/checkpoint/{eid}', exist_ok=True)
  config_str = FLAGS.flags_into_string()
  with open(f'{data_dir}/logs/{eid}/config.json', 'w') as f:
    f.write(config_str)
  with open(f'{data_dir}/checkpoint/{eid}/config.json', 'w') as f:
    f.write(config_str)

  ckpt_folder = f'{data_dir}/checkpoint/{eid}'

  checkpoint_filepath = ckpt_folder + '/epoch-{epoch:02d}/ckpt.weights.h5'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_test_loss',
      mode='min',
      save_best_only=False,
  )

  model.fit(
      trainloader,
      epochs=FLAGS.epochs,
      callbacks=[
          tb_callback,
          model_checkpoint_callback,
      ],
      validation_data=valloader,
  )

  model.evaluate(valloader)


if __name__ == '__main__':
  app.run(main)
