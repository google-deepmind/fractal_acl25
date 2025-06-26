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

"""Utils for training."""

import typing
import tensorflow as tf
import tensorflow_probability as tfp


class CustomTBCallback(tf.keras.callbacks.TensorBoard):
  """Simple callback to add hparams as a json string."""

  def __init__(
      self,
      hyperparams: str | None = None,
      log_dir: str = 'logs',
      histogram_freq: int = 0,
      write_graph: bool = True,
      write_images: bool = False,
      write_steps_per_second: bool = False,
      update_freq: str | int = 'epoch',
      profile_batch: int = 0,
      embeddings_freq: int = 0,
      embeddings_metadata: dict[str, str] | None = None,
      **kwargs: typing.Any,
  ):
    super().__init__(
        log_dir,
        histogram_freq,
        write_graph,
        write_images,
        write_steps_per_second,
        update_freq,
        profile_batch,
        embeddings_freq,
        embeddings_metadata,
        **kwargs,
    )
    self.hyperparams = hyperparams

  def set_model(self, model):
    super().set_model(model)
    if self.hyperparams is not None:
      with self._train_writer.as_default():
        tf.summary.text('hyperparameters', self.hyperparams, step=0)
      with self._val_writer.as_default():
        tf.summary.text('hyperparameters', self.hyperparams, step=0)


def BinaryCrossEntropy(y_true, y_pred):  # pylint: disable=invalid-name
  """Binary cross entropy."""
  y_pred = tf.clip_by_value(
      y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
  )
  term_0 = (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
  term_1 = y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())
  return -(term_0 + term_1)


# Cosine Simialrity Prior
def cosine_similarity_prior(
    doc_segment_cosine_similarity,
    pred_summary_segment_scores,
    num_summary_segments,
):
  """Cosine similarity prior."""
  cosine_similarity_prior_batch_samples = []
  diff = BinaryCrossEntropy(
      tf.transpose(doc_segment_cosine_similarity),
      pred_summary_segment_scores,
  ) * tf.sequence_mask(num_summary_segments, dtype=tf.float32)
  cosine_similarity_prior_batch_samples.append(
      tf.divide(
          tf.reduce_sum(diff, 1), tf.cast(num_summary_segments, tf.float32)
      )
  )
  # for sample_id in range(pred_summary_segment_scores.shape[0]):
  #   cosine_similarity_prior_batch_samples.append(tf.reduce_mean(diff[sample_id,:num_summary_segments[sample_id]]))
  return tf.reduce_mean(tf.stack(cosine_similarity_prior_batch_samples))


# Correlation Prior
def correlation_prior(
    summary_sentences_embedding_ragged,
    pred_summary_segment_scores,
    num_summary_segments,
):
  """Correlation prior."""
  correlation_prior_batch_samples = []
  mask = tf.sequence_mask(num_summary_segments)
  for sample_id in range(pred_summary_segment_scores.shape[0]):
  # pred_summary_segment_scores_sample = pred_summary_segment_scores[sample_id]
    summary_sentences_sample = summary_sentences_embedding_ragged[sample_id]
    correlation = (
        1
        + tfp.stats.correlation(
            summary_sentences_sample,
            y=None,
            sample_axis=1,
            event_axis=0,
            keepdims=False,
        )
    ) / 2
    preds = tf.boolean_mask(
        pred_summary_segment_scores[sample_id], mask[sample_id]
    )
    corr_prior = tf.reduce_sum(
        tf.abs((1 - preds) * (1 - preds[:, None]) - correlation)
        * tf.abs((preds) * (preds[:, None]) - correlation)
    )
    correlation_prior_batch_samples.append(
        corr_prior / tf.cast(num_summary_segments[sample_id], tf.float32)
    )
  return tf.reduce_mean(tf.stack(correlation_prior_batch_samples))


def loss_fn(loss_fn='mae'):  # pylint: disable=redefined-outer-name
  """Loss function."""
  loss = None
  loss_no_red = None
  if loss_fn == 'mae':
    loss = tf.keras.losses.MeanAbsoluteError()
    loss_no_red = tf.keras.losses.MeanAbsoluteError(
        reduction=tf.keras.losses.Reduction.NONE
    )
  elif loss_fn == 'mse':
    loss = tf.keras.losses.MeanSquaredError()
    loss_no_red = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE
    )
  elif loss_fn == 'bce':
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    loss_no_red = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )
  elif loss_fn == 'bradley_terry_bce':
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss_no_red = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )
  elif loss_fn == 'msle':
    loss = tf.keras.losses.MeanSquaredLogarithmicError()
    loss_no_red = tf.keras.losses.MeanSquaredLogarithmicError(
        reduction=tf.keras.losses.Reduction.NONE
    )
  return loss, loss_no_red


def get_dataloader(
    dataset_path, encoder_type='sentence-t5', batch_size=32, cv_split=1
):
  """Get dataloader."""
  ds = tf.data.Dataset.load(f'{dataset_path}/{encoder_type}_train_preference')
  cv_frac = int(len(ds) / 10)
  val_ds = ds.skip(cv_frac * (cv_split - 1)).take(cv_frac)
  train_ds = ds.take(cv_frac * (cv_split - 1)).concatenate(
      ds.skip(cv_frac * cv_split)
  )

  print('Ds length', len(ds), len(train_ds), len(val_ds))

  val_loader = val_ds.batch(
      batch_size, drop_remainder=False, num_parallel_calls=8
  )
  train_loader = train_ds.batch(
      batch_size, drop_remainder=False, num_parallel_calls=8
  )
  return train_loader, val_loader
