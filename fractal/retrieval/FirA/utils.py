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

"""Utils for FirA retrieval task."""

import typing

import tensorflow as tf
import tensorflow_probability as tfp


# Cosine Similarity Prior
def cosine_similarity_prior(
    doc_segment_cosine_similarity,
    pred_summary_segment_scores,
    num_summary_segments,
):
  """Calculates the cosine similarity prior.

  Args:
    doc_segment_cosine_similarity: Cosine similarity between document and
      segments.
    pred_summary_segment_scores: Predicted scores for summary segments.
    num_summary_segments: Number of summary segments.

  Returns:
    The cosine similarity prior.
  """
  cosine_similarity_prior_batch_samples = []
  diff = tf.abs(
      float(pred_summary_segment_scores.shape[-1] - 1)
      * tf.squeeze(tf.transpose(doc_segment_cosine_similarity), 1)
      - tf.cast(
          tf.argmax(pred_summary_segment_scores, axis=-1), dtype=tf.float32
      )
  ) * tf.sequence_mask(num_summary_segments, dtype=tf.float32)
  cosine_similarity_prior_batch_samples.append(
      tf.divide(
          tf.reduce_sum(diff, 1), tf.cast(num_summary_segments, tf.float32)
      )
  )
  return tf.reduce_mean(tf.stack(cosine_similarity_prior_batch_samples))


# Correlation Prior
def correlation_prior(
    summary_sentences_embedding_ragged,
    pred_summary_segment_scores,
    num_summary_segments,
):
  """Calculates the correlation prior.

  Args:
    summary_sentences_embedding_ragged: Ragged tensor of summary sentences
      embeddings.
    pred_summary_segment_scores: Predicted scores for summary segments.
    num_summary_segments: Number of summary segments (list).

  Returns:
    The correlation prior.
  """
  correlation_prior_batch_samples = []
  mask = tf.sequence_mask(num_summary_segments)
  inst_preds = (1 / (pred_summary_segment_scores.shape[-1] - 1)) * tf.cast(
      tf.argmax(pred_summary_segment_scores, axis=-1), dtype=tf.float32
  )
  for sample_id in range(pred_summary_segment_scores.shape[0]):
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
    preds = tf.boolean_mask(inst_preds[sample_id], mask[sample_id])
    corr_prior = tf.reduce_sum(
        tf.abs((1 - preds) * (1 - preds[:, None]) - correlation)
        * tf.abs((preds) * (preds[:, None]) - correlation)
    )
    correlation_prior_batch_samples.append(
        corr_prior / tf.cast(num_summary_segments[sample_id], tf.float32)
    )
  return tf.reduce_mean(tf.stack(correlation_prior_batch_samples))


def loss_fn(loss_fn='mae'):  # pylint: disable=redefined-outer-name
  """Calculates the loss function.

  Args:
    loss_fn: Loss function to use.

  Returns:
    The loss function.
  """
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
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss_no_red = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )
  elif loss_fn == 'msle':
    loss = tf.keras.losses.MeanSquaredLogarithmicError()
    loss_no_red = tf.keras.losses.MeanSquaredLogarithmicError(
        reduction=tf.keras.losses.Reduction.NONE
    )
  elif loss_fn == 'ce':
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss_no_red = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )
  return loss, loss_no_red


def get_dataloader(full_dataset_path, batch_size=32, cv_split=1):
  """Loads the dataset and returns the train, val, and test dataloaders.

  Args:
    full_dataset_path: Path to the full dataset.
    batch_size: Batch size for the dataloaders.
    cv_split: Cross-validation split number.

  Returns:
    Tuple of trainloader, valloader, and testloader.
  """
  ds = tf.data.Dataset.load(f'{full_dataset_path}')
  test_ds = ds.take(int(0.15 * len(ds)))
  ds = ds.skip(int(0.15 * len(ds)))
  cv_frac = int(len(ds) / 10)
  val_ds = ds.skip(cv_frac * (cv_split - 1)).take(cv_frac)
  train_ds = ds.take(cv_frac * (cv_split - 1)).concatenate(
      ds.skip(cv_frac * cv_split)
  )

  testloader = test_ds.batch(
      batch_size, drop_remainder=False, num_parallel_calls=8
  )
  valloader = val_ds.batch(
      batch_size, drop_remainder=False, num_parallel_calls=8
  )
  trainloader = train_ds.batch(
      batch_size, drop_remainder=False, num_parallel_calls=8
  )

  return trainloader, valloader, testloader


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
  y_pred = tf.clip_by_value(
      y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
  )
  term_0 = (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
  term_1 = y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())
  return -(term_0 + term_1)
