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

"""Utility functions for the MultiSpanQA model."""
import tensorflow as tf
import tensorflow_probability as tfp


def loss_fn(loss_fn='mae'):  # pylint: disable=redefined-outer-name
  """Returns the loss function."""
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
  return loss, loss_no_red


def BinaryCrossEntropy(y_true, y_pred):  # pylint: disable=invalid-name
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
  """Calculates the cosine similarity prior."""
  cosine_similarity_prior_batch_samples = []
  diff = BinaryCrossEntropy(
      tf.squeeze(tf.transpose(doc_segment_cosine_similarity), 1),
      pred_summary_segment_scores,
  ) * tf.sequence_mask(num_summary_segments, dtype=tf.float32)
  cosine_similarity_prior_batch_samples.append(
      tf.divide(
          tf.reduce_sum(diff, 1), tf.cast(num_summary_segments, tf.float32)
      )
  )
  return tf.reduce_mean(tf.stack(cosine_similarity_prior_batch_samples))


# Hybrid Prior
def hybrid_prior(
    instance_model_pred,
    pred_summary_segment_scores,
    num_summary_segments,
):
  """Calculates the hybrid prior."""
  hybrid_prior_batch_samples = []  # pylint: disable=unused-variable
  pred_summary_segment_scores = pred_summary_segment_scores - 100.0 * (
      1 - tf.sequence_mask(num_summary_segments, dtype=tf.float32)
  )
  pred_summary_segment_scores = tf.reshape(pred_summary_segment_scores, -1)
  pred_instance_scores = tf.boolean_mask(
      pred_summary_segment_scores, tf.greater(pred_summary_segment_scores, -1.0)
  )
  diff = BinaryCrossEntropy(instance_model_pred, pred_instance_scores)
  return tf.reduce_mean(diff)


# Correlation Prior
def correlation_prior(
    summary_sentences_embedding_ragged,
    pred_summary_segment_scores,
    num_summary_segments,
):
  """Calculates the correlation prior."""
  correlation_prior_batch_samples = []
  mask = tf.sequence_mask(num_summary_segments)
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


def get_dataloader(
    dataset_path, encoder_type='sentence-t5', batch_size=32, cv_split=1
):
  """Returns the dataloader."""
  dataloaders = {}
  for split in ['train', 'valid']:
    ds = tf.data.Dataset.load(f'{dataset_path}/{encoder_type}_{split}')
    if split == 'train':
      cv_frac = int(len(ds) / 10)
      val_ds = ds.skip(cv_frac * (cv_split - 1)).take(cv_frac)
      train_ds = ds.take(cv_frac * (cv_split - 1)).concatenate(
          ds.skip(cv_frac * cv_split)
      )
      trainloader = train_ds.batch(
          batch_size, drop_remainder=True, num_parallel_calls=8
      )
      valloader = val_ds.batch(
          batch_size, drop_remainder=True, num_parallel_calls=8
      )
      dataloaders['train'] = trainloader
      dataloaders['valid'] = valloader
    elif split == 'valid':
      loader = ds.batch(batch_size, drop_remainder=False, num_parallel_calls=8)
      dataloaders['test'] = loader
  return (dataloaders['train'], dataloaders['valid'], dataloaders['test'])


def get_dataloader_hybrid_prior(
    dataset_path,
    encoder_type='sentence-t5',
    batch_size=32,
    frac_disagg_bags=0.2,
    cv_split=1,
):
  """Returns the dataloader for the hybrid prior setting."""
  dataloaders = {}
  for split in ['train', 'valid']:
    ds = tf.data.Dataset.load(f'{dataset_path}/{encoder_type}_{split}')
    if split == 'train':
      disagg_loader = ds.take(int(frac_disagg_bags * len(ds))).batch(
          batch_size, drop_remainder=True, num_parallel_calls=8
      )
      ds = ds.skip(int(frac_disagg_bags * len(ds)))
      cv_frac = int(len(ds) / 10)
      val_ds = ds.skip(cv_frac * (cv_split - 1)).take(cv_frac)
      train_ds = ds.take(cv_frac * (cv_split - 1)).concatenate(
          ds.skip(cv_frac * cv_split)
      )
      trainloader = train_ds.batch(
          batch_size, drop_remainder=True, num_parallel_calls=8
      )
      valloader = val_ds.batch(
          batch_size, drop_remainder=True, num_parallel_calls=8
      )
      dataloaders['train_instances'] = disagg_loader
      dataloaders['train_bags'] = trainloader
      dataloaders['valid'] = valloader
    elif split == 'valid':
      loader = ds.batch(batch_size, drop_remainder=False, num_parallel_calls=8)
      dataloaders['test'] = loader
  return (
      dataloaders['train_instances'],
      dataloaders['train_bags'],
      dataloaders['valid'],
      dataloaders['test'],
  )


def get_dataloader_hybrid_1bags(
    dataset_path,
    encoder_type='sentence-t5',
    batch_size=32,
    frac_disagg_bags=0.2,
):
  """Returns the dataloader for the hybrid prior setting with 1 bag."""
  dataloaders = {}
  rmask = None  # pylint: disable=unused-variable
  for split in ['train', 'valid']:
    ds = tf.data.Dataset.load(f'{dataset_path}/{encoder_type}_{split}')
    ds = ds.shuffle(len(ds))
    if split == 'train':
      idx = tf.range(len(ds))
      ridx = tf.reshape(
          tf.random.shuffle(idx)[: int(frac_disagg_bags * len(ds))], (-1, 1)
      )
      rmask = tf.scatter_nd(ridx, tf.ones(len(ridx)), len(ds))
      ds = tf.data.Dataset.zip(ds, rmask)
    loader = ds.batch(batch_size, drop_remainder=False, num_parallel_calls=8)
    dataloaders[split] = loader
  return dataloaders['train'], dataloaders['valid']
