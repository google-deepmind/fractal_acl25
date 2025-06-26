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

"""Model for MultiSpanQA."""
from retrieval.MultiSpanQA.utils import correlation_prior
from retrieval.MultiSpanQA.utils import cosine_similarity_prior
from retrieval.MultiSpanQA.utils import loss_fn
from sklearn.metrics import roc_auc_score  # pylint: disable=g-importing-member
import tensorflow as tf


def get_pseudolabels_loader(model, dataloader, aggregation_fn, batch_size=32):
  """Generates a dataloader with pseudolabels."""
  pslab_ds = None
  for batch_id, batch in enumerate(dataloader):
    (
        context_segments_embedding,
        questions_embedding,
        num_context_segments,  # pylint: disable=unused-variable
        target_agg_score,  # pylint: disable=unused-variable
        target_instance_score_ragged,  # pylint: disable=unused-variable
    ) = batch
    [
        target_agg_score,
        target_instance_score,  # pylint: disable=unused-variable
        pred_context_segment_scores,
        num_context_segments,
        cosine_similarity_prior_batch_samples,  # pylint: disable=unused-variable
        correlation_prior_batch_samples,  # pylint: disable=unused-variable
        # hybrid_prior_batch_samples
    ] = model(batch)
    batch_pseudolabels = []
    for bag_id, bag_pred in enumerate(pred_context_segment_scores):
      bag_segments = num_context_segments[bag_id]
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
        context_segments_embedding,
        questions_embedding,
        num_context_segments,
        target_agg_score,
        batch_pseudolabels_ragged,
    ))
    if batch_id == 0 and pslab_ds is None:
      pslab_ds = ds_batch
    elif pslab_ds is not None:
      pslab_ds = pslab_ds.concatenate(ds_batch)
  assert pslab_ds is not None
  return pslab_ds.batch(batch_size, drop_remainder=False, num_parallel_calls=8)


class SentenceT5_MLP_Agg(tf.keras.Model):  # pylint: disable=invalid-name
  """SentenceT5 MLP Aggregation Model."""

  def __init__(
      self,
      output_dim,
      last_layer_activation,
      cosine_similarity,
      correlation,
      lambda_dllp,
      lambda_cosine_similarity,
      lambda_correlation,
      lr,
      classify,
      loss_type,
      dllp_loss_fn,
      aggregation_fn,
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
    self.lambda_dllp = lambda_dllp
    self.lambda_cosine_similarity = lambda_cosine_similarity
    self.lambda_correlation = lambda_correlation
    self.classify = classify
    self.loss_type = loss_type
    self.loss_fn = loss_fn(dllp_loss_fn)
    self.aggregation_fn = aggregation_fn

    self.lr = lr
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.test_loss = tf.keras.metrics.Mean(name='test_loss')
    self.metrics_dict = {
        'train_loss': self.train_loss,
        'test_loss': self.test_loss,
        'instance_auc': tf.keras.metrics.AUC(name='instance_auc'),
        'aggregate_auc': tf.keras.metrics.AUC(name='aggregate_auc'),
        'instance_auc_pr': tf.keras.metrics.AUC(
            name='instance_auc_pr', curve='PR'
        ),
        'aggregate_auc_pr': tf.keras.metrics.AUC(
            name='aggregate_auc_pr', curve='PR'
        ),
        'instance_mae': tf.keras.metrics.MeanAbsoluteError(name='instance_mae'),
        'aggregate_mae': tf.keras.metrics.MeanAbsoluteError(
            name='aggregate_mae'
        ),
        'instance_mse': tf.keras.metrics.MeanSquaredError(name='instance_mse'),
        'aggregate_mse': tf.keras.metrics.MeanSquaredError(
            name='aggregate_mse'
        ),
        'instance_bce': tf.keras.metrics.BinaryCrossentropy(
            name='instance_bce'
        ),
        'aggregate_bce': tf.keras.metrics.BinaryCrossentropy(
            name='aggregate_bce'
        ),
        'instance_msle': tf.keras.metrics.MeanSquaredLogarithmicError(
            name='instance_msle'
        ),
        'aggregate_msle': tf.keras.metrics.MeanSquaredLogarithmicError(
            name='aggregate_msle'
        ),
    }
    self.thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t in self.thresholds:
      self.metrics_dict[f'instance_precision_{t}'] = tf.keras.metrics.Precision(
          name=f'instance_precision_{t}', thresholds=t
      )
      self.metrics_dict[f'aggregate_precision_{t}'] = (
          tf.keras.metrics.Precision(
              name=f'aggregate_precision_{t}', thresholds=t
          )
      )
      self.metrics_dict[f'instance_recall_{t}'] = tf.keras.metrics.Recall(
          name=f'instance_recall_{t}', thresholds=t
      )
      self.metrics_dict[f'aggregate_recall_{t}'] = tf.keras.metrics.Recall(
          name=f'aggregate_recall_{t}', thresholds=t
      )
      self.metrics_dict[f'instance_accuracy_{t}'] = (
          tf.keras.metrics.BinaryAccuracy(
              name=f'instance_accuracy_{t}', threshold=t
          )
      )
      self.metrics_dict[f'aggregate_accuracy_{t}'] = (
          tf.keras.metrics.BinaryAccuracy(
              name=f'aggregate_accuracy_{t}', threshold=t
          )
      )

  def aggregate_auc_roc(
      self, target_agg_score, pred_segments_score, num_segments
  ):
    target = []
    pred = []
    agg_fn = None
    if self.aggregation_fn == 'min':
      agg_fn = tf.reduce_min
    elif self.aggregation_fn == 'max':
      agg_fn = tf.reduce_max
    for i in range(len(num_segments)):
      target.append(agg_fn(target_agg_score[i, : num_segments[i]]))
      pred.append(agg_fn(pred_segments_score[i, : num_segments[i]]))
    score = roc_auc_score(target, pred)
    return score, target, pred

  def get_instance_list(
      self, target_instance_score, pred_segments_score, num_segments
  ):
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
      self, target_instance_score, pred_segments_score, num_segments
  ):
    target = []
    pred = []
    agg_fn = None
    if self.aggregation_fn == 'min':
      agg_fn = tf.reduce_min
    elif self.aggregation_fn == 'max':
      agg_fn = tf.reduce_max
    for i in range(len(num_segments)):
      target.append(agg_fn(target_instance_score[i, : num_segments[i]]))
      pred.append(agg_fn(pred_segments_score[i, : num_segments[i]]))
    return target, pred

  def aggregate_loss(
      self, loss_fn, target_agg_score, pred_segments_score, num_segments  # pylint: disable=redefined-outer-name
  ):
    pred_agg_score = None
    if self.aggregation_fn == 'min':
      pred_agg_score = tf.reduce_min(pred_segments_score, 1)
    elif self.aggregation_fn == 'max':
      pred_agg_score = tf.reduce_max(pred_segments_score, 1)
    return loss_fn(target_agg_score, pred_agg_score)

  def instance_error(
      self,
      loss_fn,  # pylint: disable=redefined-outer-name
      target_instance_score,
      pred_segments_score,
      num_segments,
      classify=False,
  ):
    score = []

    for i in range(len(num_segments)):
      score.append(
          loss_fn(
              tf.nest.flatten(target_instance_score[i, : num_segments[i]])[0],
              tf.nest.flatten(pred_segments_score[i, : num_segments[i]])[0],
          )
      )
    return tf.stack(score)

  def call(self, batch):
    (
        context_segments_embedding,
        questions_embedding,
        num_context_segments,
        target_agg_score,
        target_instance_score_ragged,
    ) = batch
    max_context_segments = tf.reduce_max(num_context_segments)
    target_instance_score = []
    for score in range(target_instance_score_ragged.shape[0]):
      instance_scores = target_instance_score_ragged[score]
      target_instance_score.append(
          tf.concat(
              [
                  instance_scores,
                  tf.ones(
                      [max_context_segments - tf.shape(instance_scores)[0]],
                      dtype=tf.float32,
                  ),
              ],
              axis=0,
          )
      )
    target_instance_score = tf.stack(target_instance_score)
    summary_segment_scores = []
    doc_segment_cosine_similarity = []
    for segment in range(max_context_segments):
      segment_embedding = context_segments_embedding[
          :, segment : segment + 1
      ].to_tensor()
      segment_embedding = tf.squeeze(segment_embedding, axis=1)
      context_sentence_question_cosine_similarity = []
      mlp_input = tf.concat([questions_embedding, segment_embedding], axis=1)
      if self.similarity_prior:
        cosine_similarity = self.cosine_similarity(
            questions_embedding, segment_embedding
        )
        context_sentence_question_cosine_similarity.append(
            (1 + cosine_similarity) / 2
        )

      if self.similarity_prior:
        doc_segment_cosine_similarity.append(
            tf.stack(context_sentence_question_cosine_similarity)
        )
      mask = tf.cast(tf.greater(num_context_segments, segment), tf.float32)
      if self.aggregation_fn == 'max':
        mask = mask + (1 - mask) * (-100.0)
      elif self.aggregation_fn == 'min':
        mask = mask + (1 - mask) * (100.0)
      summary_segment_scores.append(
          mask * tf.squeeze(self.nli_mlp(mlp_input), axis=1)
      )
    pred_context_segment_scores = tf.stack(summary_segment_scores, axis=1)

    cosine_similarity_prior_batch_samples = None
    correlation_prior_batch_samples = None
    if self.similarity_prior:
      cosine_similarity_prior_batch_samples = cosine_similarity_prior(
          doc_segment_cosine_similarity,
          pred_context_segment_scores,
          num_context_segments,
      )
    if self.correlation_prior:
      correlation_prior_batch_samples = correlation_prior(
          context_segments_embedding,
          pred_context_segment_scores,
          num_context_segments,
      )

    return [
        target_agg_score,
        target_instance_score,
        pred_context_segment_scores,
        num_context_segments,
        cosine_similarity_prior_batch_samples,
        correlation_prior_batch_samples,
    ]

  def train_step(self, batch):
    # loss_fn = (loss, loss_no_red)
    with tf.GradientTape() as tape:
      (
          target_agg_score,
          target_instance_score,
          pred_segments_score,
          num_segments,
          cosine_similarity_prior,  # pylint: disable=redefined-outer-name
          correlation_prior,  # pylint: disable=redefined-outer-name
      ) = self(batch)
      loss = None
      if self.loss_type == 'aggregate':
        loss = self.lambda_dllp * self.aggregate_loss(
            self.loss_fn[0], target_agg_score, pred_segments_score, num_segments
        )
      elif self.loss_type == 'instance':
        loss = self.instance_error(
            self.loss_fn[1],
            target_instance_score,
            pred_segments_score,
            num_segments,
            self.classify,
        )
        loss = tf.reduce_mean(loss, 0)
      if cosine_similarity_prior is not None:
        loss += self.lambda_cosine_similarity * cosine_similarity_prior
      if correlation_prior is not None:
        loss += self.lambda_correlation * correlation_prior
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.train_loss.update_state(loss)

    batch_target_agg_list, batch_pred_agg_list = self.get_aggregate_list(
        target_instance_score, pred_segments_score, num_segments
    )
    batch_target_instance_list, batch_pred_instance_list = (
        self.get_instance_list(
            target_instance_score, pred_segments_score, num_segments
        )
    )
    metrics_result = self.eval_metrics(
        batch_target_instance_list,
        batch_pred_instance_list,
        batch_target_agg_list,
        batch_pred_agg_list,
    )

    return metrics_result

  def test_step(self, batch):
    (
        target_agg_score,
        target_instance_score,
        pred_segments_score,
        num_segments,
        cosine_similarity_prior,  # pylint: disable=redefined-outer-name, unused-variable
        correlation_prior,  # pylint: disable=redefined-outer-name, unused-variable
    ) = self(batch)
    loss = None
    if self.loss_type == 'aggregate':
      loss = self.aggregate_loss(
          self.loss_fn[0], target_agg_score, pred_segments_score, num_segments
      )
    elif self.loss_type == 'instance':
      loss = self.instance_error(
          self.loss_fn[1],
          target_instance_score,
          pred_segments_score,
          num_segments,
      )
      loss = tf.reduce_mean(loss, 0)
    self.test_loss.update_state(loss)

    batch_target_agg_list, batch_pred_agg_list = self.get_aggregate_list(
        target_instance_score, pred_segments_score, num_segments
    )
    batch_target_instance_list, batch_pred_instance_list = (
        self.get_instance_list(
            target_instance_score, pred_segments_score, num_segments
        )
    )
    metrics_result = self.eval_metrics(
        batch_target_instance_list,
        batch_pred_instance_list,
        batch_target_agg_list,
        batch_pred_agg_list,
    )
    print(metrics_result)
    return metrics_result

  @property
  def metrics(self):
    return list(self.metrics_dict.values())

  def eval_metrics(
      self,
      target_instance_list,
      pred_instance_list,
      target_agg_list,
      pred_agg_list,
  ):
    for m in self.metrics_dict:
      if m.startswith('instance'):
        self.metrics_dict[m].update_state(
            target_instance_list, pred_instance_list
        )
      elif m.startswith('aggregate'):
        self.metrics_dict[m].update_state(target_agg_list, pred_agg_list)

    return {m: self.metrics_dict[m].result() for m in self.metrics_dict.keys()}
