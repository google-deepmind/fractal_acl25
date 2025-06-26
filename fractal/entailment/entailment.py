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

"""Entailment model."""
from absl import app
from absl import flags
import numpy as np
from sklearn.metrics import accuracy_score  # pylint: disable=g-importing-member
from sklearn.metrics import auc  # pylint: disable=g-importing-member
from sklearn.metrics import precision_recall_curve  # pylint: disable=g-importing-member
from sklearn.metrics import precision_score  # pylint: disable=g-importing-member
from sklearn.metrics import recall_score  # pylint: disable=g-importing-member
from sklearn.metrics import roc_auc_score  # pylint: disable=g-importing-member
import tensorflow as tf
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import tqdm

flags.DEFINE_string(
    'data_dir',
    default='/data/preprocessed_wikicatsum',
    help='raw data location',
)
flags.DEFINE_string('encoder_type', default='sentence-t5', help='Encoder type')
flags.DEFINE_integer('output_dim', default=1, help='Output dimension')
flags.DEFINE_string('loss_fn', default='mae', help='Loss function')
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
flags.DEFINE_float(
    'frac_train_1bags', default=0.5, help='Fraction of train 1 bags'
)
flags.DEFINE_bool('correlation', default=False, help='Use correlation prior')
flags.DEFINE_bool(
    'cosine_similarity', default=False, help='Use Cosine similarity prior'
)
flags.DEFINE_bool(
    'classify', default=True, help='classification task True/False'
)
flags.DEFINE_integer('cv_split', default=1, help='CV Split')
flags.DEFINE_string('tpu', 'local', 'The BNS address of the first TPU worker.')
flags.DEFINE_string('dataset_dir', default='data/', help='Dataset directory')
flags.DEFINE_string(
    'aggregation_fn', default='min', help='Aggregation function'
)

FLAGS = flags.FLAGS

# Dataset
dataset_dir = FLAGS.dataset_dir


def get_dataloader(
    dataset_path, encoder_type='sentence-t5', batch_size=32, cv_split=1
):
  """Returns train_loader, val_loader, test_loader."""
  ds = tf.data.Dataset.load(f'{dataset_path}/{encoder_type}_train')
  val_ds = tf.data.Dataset.load(f'{dataset_path}/{encoder_type}_valid')
  test_ds = tf.data.Dataset.load(f'{dataset_path}/{encoder_type}_test')
  ds = ds.concatenate(val_ds)
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
  test_loader = test_ds.batch(
      batch_size, drop_remainder=False, num_parallel_calls=8
  )
  return train_loader, val_loader, test_loader


def get_dataloader_1bags(
    dataset_path,
    encoder_type='sentence-t5',
    batch_size=32,
    frac_train_1bags=0.5,
):
  """Returns train_loader, val_loader, test_loader."""
  dataloaders = {}
  for split in ['train', 'test', 'valid']:
    ds = tf.data.Dataset.load(f'{dataset_path}/{encoder_type}_{split}')
    ds = ds.shuffle(len(ds))
    if split == 'train':
      idx_to_remove = None
      if frac_train_1bags < 0.5:
        idx_to_remove = []
        for idx, sample in enumerate(ds):
          if tf.cast(sample[4], tf.int32) == 1:
            idx_to_remove.append(idx)
          if len(idx_to_remove) > (1 - frac_train_1bags) * len(ds):
            break
      elif split == 'train' and frac_train_1bags > 0.5:
        idx_to_remove = []
        for idx, sample in enumerate(ds):
          if tf.cast(sample[4], tf.int32) == 0:
            idx_to_remove.append(idx)
          if len(idx_to_remove) > (frac_train_1bags - 0.5) * len(ds):
            break
      if idx_to_remove is not None:
        document_splits_embedding = []
        summary_sentences_embedding = []
        num_document_splits = []
        num_summary_sentences = []
        target_agg_score = []
        target_instance_score = []
        for idx, sample in enumerate(ds):
          if idx not in idx_to_remove:
            document_splits_embedding.append(sample[0])
            summary_sentences_embedding.append(sample[1])
            num_document_splits.append(sample[2])
            num_summary_sentences.append(sample[3])
            target_agg_score.append(sample[4])
            target_instance_score.append(sample[5])

        document_splits_embedding = tf.concat(document_splits_embedding, axis=0)
        summary_sentences_embedding = tf.concat(
            summary_sentences_embedding, axis=0
        )
        target_instance_score = tf.concat(target_instance_score, axis=0)

        summary_sentences_embedding = tf.RaggedTensor.from_row_lengths(
            summary_sentences_embedding, num_summary_sentences
        )
        document_splits_embedding = tf.RaggedTensor.from_row_lengths(
            document_splits_embedding, num_document_splits
        )
        target_instance_score = tf.RaggedTensor.from_row_lengths(
            target_instance_score, num_summary_sentences
        )
        ds = tf.data.Dataset.from_tensor_slices((
            document_splits_embedding,
            summary_sentences_embedding,
            num_document_splits,
            num_summary_sentences,
            target_agg_score,
            target_instance_score,
        ))

    loader = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=8)
    dataloaders[split] = loader
  return dataloaders['train'], dataloaders['valid'], dataloaders['test']


def cosine_similarity_prior(
    doc_segment_cosine_similarity,
    pred_summary_segment_scores,
    num_summary_segments,
):
  """Calculates the cosine similarity prior."""
  cosine_similarity_prior_batch_samples = []
  diff = tf.abs(
      tf.transpose(doc_segment_cosine_similarity) - pred_summary_segment_scores
  ) * tf.sequence_mask(num_summary_segments, dtype=tf.float32)
  cosine_similarity_prior_batch_samples.append(
      tf.divide(
          tf.reduce_sum(diff, 1), tf.cast(num_summary_segments, tf.float32)
      )
  )
  return tf.reduce_mean(tf.stack(cosine_similarity_prior_batch_samples))


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


# %%
class SentenceT5_MLP_Agg(tf.keras.Model):  # pylint: disable=invalid-name
  """SentenceT5 MLP Agg model."""

  def __init__(
      self, output_dim, last_layer_activation, cosine_similarity, correlation
  ):
    super().__init__()

    self.nli_mlp = tf.keras.Sequential()
    self.emb_dim = 768 * 2
    self.nli_mlp.add(
        tf.keras.layers.Dense(self.emb_dim // 2, activation='gelu')
    )
    self.nli_mlp.add(
        tf.keras.layers.Dense(self.emb_dim // 4, activation='gelu')
    )
    self.nli_mlp.add(
        tf.keras.layers.Dense(output_dim, activation=last_layer_activation)
    )
    self.output_dim = output_dim  # dim of predicted NLI score

    self.similarity_prior = cosine_similarity  # True/False
    self.correlation_prior = correlation  # True/False

    self.cosine_similarity = tf.keras.losses.CosineSimilarity(
        reduction=tf.keras.losses.Reduction.NONE
    )

  def call(self, batch):
    (
        document_splits_embedding,
        summary_sentences_embedding,
        num_document_splits,
        num_summary_segments,
        target_agg_score,
        target_instance_score_ragged,
    ) = batch
    max_summary_segments = tf.reduce_max(num_summary_segments)
    max_document_splits = tf.reduce_max(num_document_splits)
    target_instance_score = []
    for score in range(target_instance_score_ragged.shape[0]):
      instance_scores = target_instance_score_ragged[score]
      target_instance_score.append(
          tf.concat(
              [
                  instance_scores,
                  tf.ones(
                      [max_summary_segments - instance_scores.shape[0]],
                      dtype=tf.float32,
                  ),
              ],
              axis=0,
          )
      )
    target_instance_score = tf.stack(target_instance_score)
    summary_segment_scores = []
    doc_segment_cosine_similarity = []
    for segment in range(max_summary_segments):
      segment_embedding = summary_sentences_embedding[
          :, segment : segment + 1
      ].to_tensor()
      segment_embedding = tf.squeeze(segment_embedding, axis=1)
      segment_max_score = []
      doc_split_summary_sentence_cosine_similarity = []
      for doc_split in range(max_document_splits):
        doc_split_embedding = document_splits_embedding[
            :, doc_split : doc_split + 1
        ].to_tensor()
        doc_split_embedding = tf.squeeze(doc_split_embedding, axis=1)
        mlp_input = tf.concat([doc_split_embedding, segment_embedding], axis=1)
        mask = tf.cast(tf.greater(num_document_splits, doc_split), tf.float32)
        segment_max_score.append(
            mask * tf.squeeze(self.nli_mlp(mlp_input), axis=1)
        )
        if self.similarity_prior:
          cosine_similarity = self.cosine_similarity(
              doc_split_embedding, segment_embedding
          )
          doc_split_summary_sentence_cosine_similarity.append(
              mask * (1 + cosine_similarity) / 2
          )
      segment_max_score = tf.stack(segment_max_score)
      if self.similarity_prior:
        doc_segment_cosine_similarity.append(
            tf.reduce_max(
                tf.stack(doc_split_summary_sentence_cosine_similarity), axis=0
            )
        )
      mask = tf.cast(tf.greater(num_summary_segments, 10), tf.float32)
      mask = mask + (1 - mask) * (100.0)
      summary_segment_scores.append(
          mask * tf.reduce_max(segment_max_score, axis=0)
      )
    pred_summary_segment_scores = tf.stack(summary_segment_scores, axis=1)

    cosine_similarity_prior_batch_samples = None
    correlation_prior_batch_samples = None
    if self.similarity_prior:
      cosine_similarity_prior_batch_samples = cosine_similarity_prior(
          doc_segment_cosine_similarity,
          pred_summary_segment_scores,
          num_summary_segments,
      )
    if self.correlation_prior:
      correlation_prior_batch_samples = correlation_prior(
          summary_sentences_embedding,
          pred_summary_segment_scores,
          num_summary_segments,
      )

    return (
        target_agg_score,
        target_instance_score,
        pred_summary_segment_scores,
        num_summary_segments,
        cosine_similarity_prior_batch_samples,
        correlation_prior_batch_samples,
    )


def loss_fn(loss_fn='mae'):  # pylint: disable=redefined-outer-name
  """Returns the loss function.

  Args:
    loss_fn: Loss function type.

  Returns:
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
  return loss, loss_no_red


def aggregate_loss(
    loss_fn, target_agg_score, pred_segments_score, num_segments  # pylint: disable=redefined-outer-name, unused-argument
):
  # print(target_agg_score.shape, pred_segments_score.shape)
  # pred_agg_score = tf.divide(tf.reduce_sum(pred_segments_score, 1),
  #                                       tf.cast(num_segments, tf.float32))
  pred_agg_score = tf.reduce_min(pred_segments_score, 1)
  return loss_fn(target_agg_score, pred_agg_score)


def instance_error(
    loss_fn,  # pylint: disable=redefined-outer-name
    target_instance_score,
    pred_segments_score,
    num_segments,
    classify=False,  # pylint: disable=unused-argument
):
  """Calculates the instance error."""
  score = []
  for i in range(len(num_segments)):
    score.append(
        loss_fn(
            tf.nest.flatten(target_instance_score[i, : num_segments[i]]),
            tf.nest.flatten(pred_segments_score[i, : num_segments[i]]),
        )
    )
  return tf.concat(score, axis=0)


def instance_auc_roc(target_instance_score, pred_segments_score, num_segments):
  target = []
  pred = []
  for i in range(len(num_segments)):
    target += tf.nest.flatten(target_instance_score[i, : num_segments[i]])
    pred += tf.nest.flatten(pred_segments_score[i, : num_segments[i]])
  score = roc_auc_score(tf.concat(target, axis=0), tf.concat(pred, axis=0))
  return score


def aggregate_auc_roc(
    target_agg_score, pred_segments_score, num_segments, aggregation_fn='min'
):
  """Calculates the aggregate AUC ROC.

  Args:
    target_agg_score: Target aggregate score
    pred_segments_score: Predicted segments score
    num_segments: Number of segments
    aggregation_fn: Aggregation function

  Returns:
  """
  target = []
  pred = []
  agg_fn = None
  if aggregation_fn == 'min':
    agg_fn = tf.reduce_min
  for i in range(len(num_segments)):
    target.append(agg_fn(target_agg_score[i, : num_segments[i]]))
    pred.append(agg_fn(pred_segments_score[i, : num_segments[i]]))
  score = roc_auc_score(target, pred)
  return score, target, pred


def constant_baseline_error(loss_fn, target_agg_score):  # pylint: disable=redefined-outer-name
  return min([
      tf.keras.backend.get_value(
          loss_fn(
              target_agg_score,
              tf.constant(i / 10, dtype=tf.float32)
              * tf.ones(tf.shape(target_agg_score)),
          )
      )
      for i in range(10)
  ])


def train_step(
    batch,
    model,
    optimizer,
    loss_fn,  # pylint: disable=redefined-outer-name
    loss_type,
    classify,
    lambda_dllp,
    lambda_cosine_similarity,
    lambda_correlation,
):
  """Performs a single training step.

  Args:
    batch: A batch of training data.
    model: The model to train.
    optimizer: The optimizer to use.
    loss_fn: The loss function to use.
    loss_type: The type of loss to use.
    classify: Whether to perform classification.
    lambda_dllp: The weight for the DLLP loss.
    lambda_cosine_similarity: The weight for the cosine similarity loss.
    lambda_correlation: The weight for the correlation loss.

  Returns:
    The loss.
  """
  # loss_fn = (loss, loss_no_red)
  with tf.GradientTape() as tape:
    (
        target_agg_score,
        target_instance_score,
        pred_segments_score,
        num_segments,
        cosine_similarity_prior,  # pylint: disable=redefined-outer-name
        correlation_prior,  # pylint: disable=redefined-outer-name
    ) = model(batch)
    loss = None
    if loss_type == 'aggregate':
      loss = lambda_dllp * aggregate_loss(
          loss_fn[0], target_agg_score, pred_segments_score, num_segments
      )
    elif loss_type == 'instance':
      loss = instance_error(
          loss_fn[1],
          target_instance_score,
          pred_segments_score,
          num_segments,
          classify,
      )
      loss = tf.reduce_mean(loss, 0)
    if cosine_similarity_prior is not None:
      loss += lambda_cosine_similarity * cosine_similarity_prior
    if correlation_prior is not None:
      loss += lambda_correlation * correlation_prior
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  # instance_mae = instance_mae(loss_fn, target_instance_score,
  # pred_segments_score)
  return loss


def test_step(
    batch, model, loss_fn, loss_type, classify, calculate_metrics=False  # pylint: disable=redefined-outer-name
):
  """Test step."""
  # loss_fn = (loss, loss_no_red)
  (
      target_agg_score,
      target_instance_score,
      pred_segments_score,
      num_segments,
      cosine_similarity_prior,  # pylint: disable=redefined-outer-name, unused-variable
      correlation_prior,  # pylint: disable=redefined-outer-name, unused-variable
  ) = model(batch)
  loss = None
  if loss_type == 'aggregate':
    loss = aggregate_loss(
        loss_fn[0], target_agg_score, pred_segments_score, num_segments
    )
  elif loss_type == 'instance':
    loss = instance_error(
        loss_fn[1], target_instance_score, pred_segments_score, num_segments
    )
    loss = tf.reduce_mean(loss, 0)
  ar = None
  if classify:
    ar = instance_auc_roc(
        target_instance_score, pred_segments_score, num_segments
    )

  if calculate_metrics:
    return target_instance_score, pred_segments_score, num_segments
  else:
    return (
        loss,
        tf.reduce_min(
            instance_error(
                loss_fn[1],
                target_instance_score,
                pred_segments_score,
                num_segments,
            ),
            0,
        ),
        constant_baseline_error(loss_fn[0], target_agg_score),
        constant_baseline_error(loss_fn[0], target_instance_score),
        ar,
    )


def get_instance_list(target_instance_score, pred_segments_score, num_segments):
  """get_instance_list.

  Args:
    target_instance_score:
    pred_segments_score:
    num_segments:

  Returns:
  """
  target = []
  pred = []
  for i in range(len(num_segments)):
    target += (
        tf.nest.flatten(target_instance_score[i, : num_segments[i]])[0]
        .numpy()
        .tolist()
    )
    pred += (
        tf.nest.flatten(pred_segments_score[i, : num_segments[i]])[0]
        .numpy()
        .tolist()
    )
  return target, pred


def get_aggregate_list(
    target_instance_score, pred_segments_score, num_segments
):
  _, target_agg_list, pred_agg_list = aggregate_auc_roc(
      target_instance_score,
      pred_segments_score,
      num_segments,
      aggregation_fn='min',
  )
  return target_agg_list, pred_agg_list


def eval_metrics(
    target_instance_list, pred_segments_list, target_agg_list, pred_agg_list
):
  """Calculates the evaluation metrics."""
  instance_auc = roc_auc_score(target_instance_list, pred_segments_list)
  aggregate_auc = roc_auc_score(target_agg_list, pred_agg_list)
  instance_mae = loss_fn('mae')[0](target_instance_list, pred_segments_list)  # pylint: disable=unused-variable
  aggregate_mae = loss_fn('mae')[0](target_agg_list, pred_agg_list)  # pylint: disable=unused-variable
  instance_mse = loss_fn('mse')[0](target_instance_list, pred_segments_list)  # pylint: disable=unused-variable
  aggregate_mse = loss_fn('mse')[0](target_agg_list, pred_agg_list)  # pylint: disable=unused-variable
  instance_bce = loss_fn('bce')[0](target_instance_list, pred_segments_list)
  aggregate_bce = loss_fn('bce')[0](target_agg_list, pred_agg_list)
  target_instance_list = np.array(target_instance_list)
  target_agg_list = np.array(target_agg_list)
  pred_segments_list = np.array(pred_segments_list)
  pred_agg_list = np.array(pred_agg_list)
  instance_accuracy, instance_recall, instance_precision = {}, {}, {}
  aggregate_accuracy, aggregate_recall, aggregate_precision = {}, {}, {}
  for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:  # pylint: disable=redefined-outer-name
    instance_accuracy[threshold] = accuracy_score(
        target_instance_list, (pred_segments_list > threshold)
    )
    aggregate_accuracy[threshold] = accuracy_score(
        target_agg_list, (pred_agg_list > threshold)
    )
    instance_recall[threshold] = recall_score(
        target_instance_list, (pred_segments_list > threshold)
    )
    aggregate_recall[threshold] = recall_score(
        target_agg_list, (pred_agg_list > threshold)
    )
    instance_precision[threshold] = precision_score(
        target_instance_list, (pred_segments_list > threshold)
    )
    aggregate_precision[threshold] = precision_score(
        target_agg_list, (pred_agg_list > threshold)
    )

  inst_precision, inst_recall, _ = precision_recall_curve(
      target_instance_list, pred_segments_list
  )
  instance_auc_pr = auc(inst_recall, inst_precision)

  agg_precision, agg_recall, _ = precision_recall_curve(
      target_agg_list, pred_agg_list
  )
  agg_auc_pr = auc(agg_recall, agg_precision)

  metrics = {
      'instance_auc': instance_auc,
      'aggregate_auc': aggregate_auc,
      'instance_bce': instance_bce,
      'aggregate_bce': aggregate_bce,
      'instance_auc_pr': instance_auc_pr,
      'agg_auc_pr': agg_auc_pr,
  }
  for threshold in instance_precision:
    metrics[f'instance_precision {threshold}'] = instance_precision[threshold]
    metrics[f'instance_recall {threshold}'] = instance_recall[threshold]
    metrics[f'agg_precision {threshold}'] = aggregate_precision[threshold]
    metrics[f'agg_recall {threshold}'] = aggregate_recall[threshold]
    metrics[f'agg_accuracy {threshold}'] = aggregate_accuracy[threshold]
    metrics[f'instance_accuracy {threshold}'] = instance_accuracy[threshold]

  return metrics


def evaluate_model_metrics(model, loss, trainloader, testloader):
  """Calculates the evaluation metrics.

  Args:
    model: The model to evaluate.
    loss: The loss function to use.
    trainloader: The training data loader.
    testloader: The test data loader.

  Returns:
    A dictionary of evaluation metrics.
  """
  split_metrics = {}
  for split in ['train', 'test']:
    target_agg_list = []
    pred_agg_list = []
    target_instance_list = []
    pred_instance_list = []

    if split == 'train':
      loader = trainloader
    else:
      loader = testloader
    for batch in tqdm.tqdm(loader, leave=False):
      (
          batch_target_instance_score,
          batch_pred_segments_score,
          batch_num_segments,
      ) = test_step(
          batch,
          model,
          loss,
          FLAGS.loss_type,
          FLAGS.classify,
          calculate_metrics=True,
      )
      batch_target_agg_list, batch_pred_agg_list = get_aggregate_list(
          batch_target_instance_score,
          batch_pred_segments_score,
          batch_num_segments,
      )
      batch_target_instance_list, batch_pred_instance_list = get_instance_list(
          batch_target_instance_score,
          batch_pred_segments_score,
          batch_num_segments,
      )
      target_agg_list += batch_target_agg_list
      pred_agg_list += batch_pred_agg_list
      target_instance_list += batch_target_instance_list
      pred_instance_list += batch_pred_instance_list
    metrics = eval_metrics(
        target_instance_list, pred_instance_list, target_agg_list, pred_agg_list
    )
    print(split)
    print(metrics)
    split_metrics[split] = metrics
  return split_metrics


def get_pseudolabels_loader(model, dataloader, aggregation_fn, batch_size=32):
  """Generates pseudolabels for the given dataloader.

  Args:
    model: The model to use for prediction.
    dataloader: The dataloader to use for pseudolabel generation.
    aggregation_fn: The aggregation function to use.
    batch_size: The batch size for pseudolabels loader.

  Returns:
    A dataloader with pseudolabels.
  """
  pslab_ds = None
  for batch_id, batch in enumerate(dataloader):
    (
        document_splits_embedding,
        summary_sentences_embedding,
        num_document_splits,
        num_summary_segments,  # pylint: disable=unused-variable
        target_agg_score,  # pylint: disable=unused-variable
        target_instance_score_ragged,  # pylint: disable=unused-variable
    ) = batch
    [
        target_agg_score,
        target_instance_score,  # pylint: disable=unused-variable
        pred_context_segment_scores,
        num_summary_segments,
        cosine_similarity_prior_batch_samples,  # pylint: disable=unused-variable
        correlation_prior_batch_samples,  # pylint: disable=unused-variable
        # hybrid_prior_batch_samples
    ] = model(batch)
    batch_pseudolabels = []
    for bag_id, bag_pred in enumerate(pred_context_segment_scores):
      bag_segments = num_summary_segments[bag_id]
      bag_agg_target = target_agg_score[bag_id]
      bag_instances_pred = bag_pred[:bag_segments]
      bag_instances_pseudolabels = tf.cast(bag_instances_pred > 0.5, tf.float32)
      if aggregation_fn == 'max':
        bag_agg_pred = tf.reduce_max(bag_instances_pseudolabels)
        if bag_agg_pred != bag_agg_target:
          if bag_agg_target == 0.0:
            bag_instances_pseudolabels = tf.constant(
                [0.0] * int(bag_segments), dtype=tf.float32
            )
          else:
            flip_label_idx = [[tf.argmax(bag_instances_pred)]]
            bag_instances_pseudolabels = tf.tensor_scatter_nd_update(
                bag_instances_pseudolabels, flip_label_idx, [1.0]
            )
      elif aggregation_fn == 'min':
        bag_agg_pred = tf.reduce_min(bag_instances_pseudolabels)
        if bag_agg_pred != bag_agg_target:
          if bag_agg_target == 1.0:
            bag_instances_pseudolabels = tf.constant(
                [1.0] * int(bag_segments), dtype=tf.float32
            )
          else:
            flip_label_idx = [[tf.argmin(bag_instances_pred)]]
            bag_instances_pseudolabels = tf.tensor_scatter_nd_update(
                bag_instances_pseudolabels, flip_label_idx, [0.0]
            )
      batch_pseudolabels.append(bag_instances_pseudolabels)
    batch_pseudolabels_ragged = tf.ragged.stack(batch_pseudolabels)
    ds_batch = tf.data.Dataset.from_tensor_slices((
        document_splits_embedding,
        summary_sentences_embedding,
        num_document_splits,
        num_summary_segments,
        target_agg_score,
        batch_pseudolabels_ragged,
    ))

    if batch_id == 0 and pslab_ds is None:
      pslab_ds = ds_batch
    elif pslab_ds is not None:
      pslab_ds = pslab_ds.concatenate(ds_batch)
  assert pslab_ds is not None
  return pslab_ds.batch(batch_size, drop_remainder=False, num_parallel_calls=8)


metrics_to_log = [
    'instance_auc',
    'aggregate_auc',
    'instance_bce',
    'aggregate_bce',
    'instance_auc_pr',
    'agg_auc_pr',
    'train_loss',
    'val_loss',
]

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
  metrics_to_log.append(f'instance_precision {threshold}')
  metrics_to_log.append(f'instance_recall {threshold}')
  metrics_to_log.append(f'agg_precision {threshold}')
  metrics_to_log.append(f'agg_recall {threshold}')
  metrics_to_log.append(f'agg_accuracy {threshold}')
  metrics_to_log.append(f'instance_accuracy {threshold}')


# %%
def main():
  print(FLAGS)
  model = None

  if FLAGS.encoder_type == 'sentence-t5':
    model = SentenceT5_MLP_Agg(
        output_dim=FLAGS.output_dim,
        last_layer_activation=FLAGS.last_layer_activation,
        correlation=FLAGS.correlation,
        cosine_similarity=FLAGS.cosine_similarity,
    )
  assert model is not None
  loss = loss_fn(FLAGS.loss_fn)
  trainloader, valloader, testloader = get_dataloader(
      dataset_dir,
      FLAGS.encoder_type,
      batch_size=FLAGS.batch_size,
      cv_split=FLAGS.cv_split,
  )
  # trainloader, valloader, testloader = get_dataloader_1bags(
  #     dataset_dir, FLAGS.encoder_type, batch_size=FLAGS.batch_size,
  #     frac_train_1bags=FLAGS.frac_train_1bags
  # )
  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  val_loss = tf.keras.metrics.Mean(name='val_loss')
  test_loss = tf.keras.metrics.Mean(name='test_loss')
  train_loss_epoch = []
  val_loss_epoch = []
  print('starting prior model training')
  for epoch in tqdm.tqdm(range(FLAGS.epochs)):
    print(f'Epoch {epoch}')
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    val_loss.reset_states()
    for batch in tqdm.tqdm(trainloader, leave=False):
      epoch_loss = train_step(
          batch,
          model,
          optimizer,
          loss,
          FLAGS.loss_type,
          FLAGS.classify,
          FLAGS.lambda_dllp,
          FLAGS.lambda_cosine_similarity,
          FLAGS.lambda_correlation,
      )
      train_loss(epoch_loss)
    train_loss_epoch.append(train_loss.result())
    for batch in valloader:
      val_loss(
          test_step(batch, model, loss, FLAGS.loss_type, FLAGS.classify)[0]
      )
    val_loss_epoch.append(val_loss.result())
    print(
        f'Epoch {epoch + 1}, '
        f'Train Loss: {train_loss.result()}, '
        f'Validation Loss: {val_loss.result()}, '
    )
    print(f'train_loss at epoch {epoch}', train_loss.result())
    print(f'val_loss at epoch {epoch}', val_loss.result())
    if epoch % 10 == 0 or epoch == FLAGS.epochs - 1:
      print('Evaluation Metrics:')
      metrics = evaluate_model_metrics(model, loss, trainloader, testloader)
      print(metrics['test'])

  test_agg_loss = []
  instance_error_list = []
  constant_baseline_agg = []
  constant_baseline_instance = []
  auc_roc = []

  # pslab training
  if FLAGS.loss_type == 'aggregate':
    print('Starting pslab training')
    psl_model = SentenceT5_MLP_Agg(
        output_dim=FLAGS.output_dim,
        last_layer_activation=FLAGS.last_layer_activation,
        correlation=False,
        cosine_similarity=False,
    )
    pslab_trainloader = get_pseudolabels_loader(
        model, trainloader, FLAGS.aggregation_fn, FLAGS.batch_size
    )
    psl_optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    train_loss_epoch = []
    val_loss_epoch = []
    for epoch in tqdm.tqdm(range(FLAGS.epochs)):
      print(f'Epoch {epoch}')
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      val_loss.reset_states()
      for batch in tqdm.tqdm(pslab_trainloader, leave=False):
        epoch_loss = train_step(
            batch,
            psl_model,
            psl_optimizer,
            loss,
            'instance',
            True,
            1.0,
            0.0,
            0.0,
        )
        train_loss(epoch_loss)
      train_loss_epoch.append(train_loss.result())
      for batch in valloader:
        val_loss(test_step(batch, psl_model, loss, 'instance', True)[0])
      val_loss_epoch.append(val_loss.result())
      print(
          f'Epoch {epoch + 1}, '
          f'Train Loss: {train_loss.result()}, '
          f'Validation Loss: {val_loss.result()}, '
      )
      print(f'train_loss at epoch {epoch}', train_loss.result())
      print(f'val_loss at epoch {epoch}', val_loss.result())
      if epoch % 10 == 0 or epoch == FLAGS.epochs - 1:
        print('PSLAB Evaluation Metrics:')
        metrics = evaluate_model_metrics(
            psl_model, loss, pslab_trainloader, testloader
        )
        print(metrics['test'])
    prior_model = model  # pylint: disable=unused-variable
    model = psl_model

  # We can evaluate prior_model also (similar to this)
  if FLAGS.classify:
    auc_roc = []
  for batch in testloader:
    t, i, ca, ci, aucr = test_step(  # pylint: disable=unbalanced-tuple-unpacking
        batch, model, loss, FLAGS.loss_type, FLAGS.classify
    )
    test_agg_loss.append(t)
    instance_error_list.append(i)
    constant_baseline_agg.append(ca)
    constant_baseline_instance.append(ci)
    test_loss(t)
    if FLAGS.classify:
      auc_roc.append(aucr)
  # print(
  # f'Test Agg Loss: {test_loss.result()}, Tess Agg avg:
  # {sum(test_agg_loss)/len(test_agg_loss)} '
  # )
  # print(
  # f'''Test Loss: {sum(test_agg_loss)/len(test_agg_loss)}
  #     Instance: {sum(instance_error_list)/len(instance_error_list)}
  #     Constant Baseline Agg:
  #     {sum(constant_baseline_agg)/len(constant_baseline_agg)}
  #     Constant Baseline Instance:
  #     {sum(constant_baseline_instance)/len(constant_baseline_instance)} '''
  # )

  if FLAGS.classify:
    print(f'AUC Roc: {sum(auc_roc)/len(auc_roc)} ')


if __name__ == '__main__':
  app.run(main)
