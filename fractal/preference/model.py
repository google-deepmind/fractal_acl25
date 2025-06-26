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

"""FRACTAL model."""

from preference.utils import correlation_prior
from preference.utils import cosine_similarity_prior
from preference.utils import loss_fn
from sklearn.metrics import roc_auc_score  # pylint: disable=g-importing-member
import tensorflow as tf


class SentenceT5_MLP_Preference(tf.keras.Model):  # pylint: disable=invalid-name
  """FRACTAL model."""

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
      data_dir,
  ):

    super().__init__()

    self.data_dir = data_dir
    self.nli_mlp = tf.keras.Sequential()
    self.emb_dim = 768 * 3
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
    self.loss_fn_name = dllp_loss_fn

    self.lr = lr
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
    self.train_loss = tf.keras.metrics.Mean(name='aggregate_train_loss')
    self.test_loss = tf.keras.metrics.Mean(name='aggregate_test_loss')
    self.metrics_dict = {
        'aggregate_train_loss': self.train_loss,
        'aggregate_test_loss': self.test_loss,
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
        'instance_bce': tf.keras.metrics.CategoricalCrossentropy(
            name='instance_bce'
        ),
        'aggregate_bce': tf.keras.metrics.CategoricalCrossentropy(
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

  def get_fine_dataloader(self, split='dev'):
    encoder_type = 'sentence-t5'
    ds = tf.data.Dataset.load(
        f'{self.data_dir}/{encoder_type}_{split}_finegrained'
    )
    ds = ds.shuffle(len(ds))
    batch_size = len(ds)
    return ds.batch(batch_size, drop_remainder=True, num_parallel_calls=8)

  def aggregate_auc_roc(
      self, target_agg_score, pred_segments_score, num_segments
  ):
    target = []
    pred = []
    agg_fn = None  # pylint: disable=unused-variable
    if self.aggregation_fn == 'min':
      for i, num_segments_i in enumerate(num_segments):  # pylint: disable=unused-variable
        target.append(tf.reduce_min(target_agg_score[i, : num_segments[i]]))
        pred.append(tf.reduce_min(pred_segments_score[i, : num_segments[i]]))
    elif self.aggregation_fn == 'avg':
      for i, num_segments_i in enumerate(num_segments):  # pylint: disable=unused-variable
        target.append(
            tf.divide(
                tf.reduce_sum(target_agg_score[i, : num_segments[i]]),
                tf.cast(num_segments[i], tf.float32),
            )
        )
        pred.append(
            tf.divide(
                tf.reduce_sum(pred_segments_score[i, : num_segments[i]]),
                tf.cast(num_segments[i], tf.float32),
            )
        )
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

  def aggregate_scores(self, pred_segments_score, num_segments):
    pred_agg_score = None
    if self.aggregation_fn == 'min':
      pred_agg_score = tf.reduce_min(pred_segments_score, 1)
    elif self.aggregation_fn == 'avg':
      pred_agg_score = tf.divide(
          tf.reduce_sum(pred_segments_score, 1),
          tf.cast(num_segments, tf.float32),
      )
    return pred_agg_score

  def remove_ties(self, pred1_agg_score, pred2_agg_score, target_pref_12_score):
    # Remove ties
    non_ties_idx = tf.reshape(tf.where(target_pref_12_score != 0), -1).numpy()
    pred1_agg_score = tf.stack([pred1_agg_score[i] for i in non_ties_idx])
    pred2_agg_score = tf.stack([pred2_agg_score[i] for i in non_ties_idx])
    target_agg_score = tf.stack(
        [target_pref_12_score[i] - 1 for i in non_ties_idx]
    )
    return pred1_agg_score, pred2_agg_score, target_agg_score

  def call(self, batch):
    (
        passage_splits_embedding,
        question_embedding,
        response_sentences_embedding,
        num_passage_splits,
        num_response_segments,
    ) = batch
    max_response_segments = tf.reduce_max(num_response_segments)
    max_passage_splits = tf.reduce_max(num_passage_splits)
    response_segment_scores = []
    doc_segment_cosine_similarity = []
    for segment in range(max_response_segments):
      segment_embedding = response_sentences_embedding[
          :, segment : segment + 1
      ].to_tensor()
      segment_embedding = tf.squeeze(segment_embedding, axis=1)
      segment_max_score = []
      passage_split_response_sentence_cosine_similarity = []
      for doc_split in range(max_passage_splits):
        doc_split_embedding = passage_splits_embedding[
            :, doc_split : doc_split + 1
        ].to_tensor()
        doc_split_embedding = tf.squeeze(doc_split_embedding, axis=1)
        mlp_input = tf.concat(
            [doc_split_embedding, question_embedding, segment_embedding], axis=1
        )
        mask = tf.cast(tf.greater(num_passage_splits, doc_split), tf.float32)
        segment_max_score.append(
            mask * tf.squeeze(self.nli_mlp(mlp_input), axis=1)
        )
        if self.similarity_prior:
          cosine_similarity = self.cosine_similarity(
              doc_split_embedding, segment_embedding
          )
          passage_split_response_sentence_cosine_similarity.append(
              mask * (1 + cosine_similarity) / 2
          )
      segment_max_score = tf.stack(segment_max_score)
      if self.similarity_prior:
        doc_segment_cosine_similarity.append(
            tf.reduce_max(
                tf.stack(passage_split_response_sentence_cosine_similarity),
                axis=0,
            )
        )
      mask = tf.cast(tf.greater(num_response_segments, segment), tf.float32)
      if self.aggregation_fn == 'min':
        mask = mask + (1 - mask) * (100.0)
      response_segment_scores.append(
          mask * tf.reduce_max(segment_max_score, axis=0)
      )
    pred_response_segment_scores = tf.stack(response_segment_scores, axis=1)

    cosine_similarity_prior_batch_samples = None
    correlation_prior_batch_samples = None
    if self.similarity_prior:
      cosine_similarity_prior_batch_samples = cosine_similarity_prior(
          doc_segment_cosine_similarity,
          pred_response_segment_scores,
          num_response_segments,
      )
    if self.correlation_prior:
      correlation_prior_batch_samples = correlation_prior(
          response_sentences_embedding,
          pred_response_segment_scores,
          num_response_segments,
      )

    return [
        pred_response_segment_scores,
        num_response_segments,
        cosine_similarity_prior_batch_samples,
        correlation_prior_batch_samples,
    ]

  def train_step(self, batch):
    # loss_fn = (loss, loss_no_red)
    with tf.GradientTape() as tape:
      (
          passage_splits_embedding,
          question_embedding,
          pred1_embedding,
          pred2_embedding,
          num_passage_splits,
          num_instance_pred1,
          num_instance_pred2,
          target_pref_12,
      ) = batch
      (
          pred1_segments_score,
          num_instance_pred1,
          cosine_similarity_prior_pred1,
          correlation_prior_pred1,
      ) = self((
          passage_splits_embedding,
          question_embedding,
          pred1_embedding,
          num_passage_splits,
          num_instance_pred1,
      ))
      (
          pred2_segments_score,
          num_instance_pred2,
          cosine_similarity_prior_pred2,
          correlation_prior_pred2,
      ) = self((
          passage_splits_embedding,
          question_embedding,
          pred2_embedding,
          num_passage_splits,
          num_instance_pred2,
      ))
      pred_pref_labels = []
      target_pref_labels = []
      loss = None
      if self.loss_type == 'preference':
        pred1_agg_score = self.aggregate_scores(
            pred1_segments_score, num_instance_pred1
        )
        pred2_agg_score = self.aggregate_scores(
            pred2_segments_score, num_instance_pred2
        )
        if self.loss_fn_name == 'bradley_terry':
          pred1_agg_score, pred2_agg_score, target_pref_labels = (
              self.remove_ties(pred1_agg_score, pred2_agg_score, target_pref_12)
          )
          pred_pref_labels = tf.math.log(
              tf.divide(pred2_agg_score, pred1_agg_score + 1e-6)
          )
          loss = self.lambda_dllp * (target_pref_labels * pred_pref_labels)

        elif self.loss_fn_name != 'bradley_terry_bce':
          pred_pref_labels = tf.sign(pred2_agg_score - pred1_agg_score)
          target_pref_labels = tf.cast(
              tf.equal(target_pref_12, 2), tf.float32
          ) - tf.cast(tf.equal(target_pref_12, 1), tf.float32)

          loss = self.lambda_dllp * self.loss_fn[0](
              target_pref_labels, pred_pref_labels
          )
        elif self.loss_fn_name == 'bradley_terry_bce':
          pred1_agg_score, pred2_agg_score, target_pref_labels = (
              self.remove_ties(pred1_agg_score, pred2_agg_score, target_pref_12)
          )
          pred_pref_labels = tf.divide(
              pred2_agg_score, pred1_agg_score + pred2_agg_score + 1e-6
          )

          loss = self.lambda_dllp * self.loss_fn[0](
              target_pref_labels, pred_pref_labels
          )

        if (
            cosine_similarity_prior_pred1 is not None
            and cosine_similarity_prior_pred2 is not None
        ):
          loss += self.lambda_cosine_similarity * (
              cosine_similarity_prior_pred1 + cosine_similarity_prior_pred2
          )
        if (
            correlation_prior_pred1 is not None
            and correlation_prior_pred2 is not None
        ):
          loss += self.lambda_correlation * (
              correlation_prior_pred1 + correlation_prior_pred2
          )
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.train_loss.update_state(loss)

    metrics_result = self.eval_metrics(
        target_pref_labels, pred_pref_labels, starts_with='aggregate'
    )

    return metrics_result

  def test_step(self, batch):
    (
        passage_splits_embedding,
        question_embedding,
        pred1_embedding,
        pred2_embedding,
        num_passage_splits,
        num_instance_pred1,
        num_instance_pred2,
        target_pref_12,
    ) = batch
    (
        pred1_segments_score,
        num_instance_pred1,
        cosine_similarity_prior_pred1,  # pylint: disable=unused-variable
        correlation_prior_pred1,  # pylint: disable=unused-variable
    ) = self((
        passage_splits_embedding,
        question_embedding,
        pred1_embedding,
        num_passage_splits,
        num_instance_pred1,
    ))
    (
        pred2_segments_score,
        num_instance_pred2,
        cosine_similarity_prior_pred2,  # pylint: disable=unused-variable
        correlation_prior_pred2,  # pylint: disable=unused-variable
    ) = self((
        passage_splits_embedding,
        question_embedding,
        pred2_embedding,
        num_passage_splits,
        num_instance_pred2,
    ))
    pred_pref_labels = []
    target_pref_labels = []
    loss = None
    if self.loss_type == 'preference':
      pred1_agg_score = self.aggregate_scores(
          pred1_segments_score, num_instance_pred1
      )
      pred2_agg_score = self.aggregate_scores(
          pred2_segments_score, num_instance_pred2
      )
      if self.loss_fn_name == 'bradley_terry':
        pred1_agg_score, pred2_agg_score, target_pref_labels = self.remove_ties(
            pred1_agg_score, pred2_agg_score, target_pref_12
        )
        pred_pref_labels = tf.math.log(
            tf.divide(pred2_agg_score, pred1_agg_score + 1e-6)
        )
        loss = self.lambda_dllp * (target_pref_labels * pred_pref_labels)
      if self.loss_fn_name == 'bce':
        pred_pref_labels = tf.sign(pred2_agg_score - pred1_agg_score)
        target_pref_labels = tf.cast(
            tf.equal(target_pref_12, 2), tf.float32
        ) - tf.cast(tf.equal(target_pref_12, 1), tf.float32)

        loss = self.lambda_dllp * self.loss_fn[0](
            target_pref_labels, pred_pref_labels
        )
      elif self.loss_fn_name == 'bradley_terry_bce':
        pred1_agg_score, pred2_agg_score, target_pref_labels = self.remove_ties(
            pred1_agg_score, pred2_agg_score, target_pref_12
        )
        pred_pref_labels = tf.divide(
            pred2_agg_score, pred1_agg_score + pred2_agg_score
        )

        loss = self.lambda_dllp * self.loss_fn[0](
            target_pref_labels, pred_pref_labels
        )
      self.test_loss.update_state(loss)

    pref_metrics_result = self.eval_metrics(
        target_pref_labels, pred_pref_labels, starts_with='aggregate'
    )

    fine_ds = self.get_fine_dataloader()
    target_instance_list, pred_instance_list = [], []
    for fine_batch in fine_ds:
      target_instance, pred_instance = self.fine_grained_predictions(fine_batch)
      target_instance_list += target_instance
      pred_instance_list += pred_instance
    instance_metric_result = self.eval_metrics(
        target_instance_list, pred_instance_list, starts_with='instance'
    )
    print('Validation pref metrics', pref_metrics_result)
    print('Test instance metrics', instance_metric_result)
    return {**pref_metrics_result, **instance_metric_result}

  def fine_grained_predictions(self, batch):
    (
        passage_splits_embedding,
        questions_embedding,
        pred_embedding,
        num_passage_splits,
        num_instance_pred,
        target_aggregate_labels,  # pylint: disable=unused-variable
        target_instance_score_ragged,
    ) = batch
    (
        pred_segments_score,
        num_instance_pred,
        cosine_similarity_prior_pred,  # pylint: disable=unused-variable
        correlation_prior_pred,  # pylint: disable=unused-variable
    ) = self((
        passage_splits_embedding,
        questions_embedding,
        pred_embedding,
        num_passage_splits,
        num_instance_pred,
    ))
    target_instance_score = []
    max_response_segments = tf.reduce_max(num_instance_pred)
    for score in range(target_instance_score_ragged.shape[0]):
      instance_scores = target_instance_score_ragged[score]
      target_instance_score.append(
          tf.concat(
              [
                  instance_scores,
                  tf.ones(
                      [max_response_segments - tf.shape(instance_scores)[0]],
                      dtype=tf.float32,
                  ),
              ],
              axis=0,
          )
      )
    target_instance_score = tf.stack(target_instance_score)
    target_list, pred_list = self.get_instance_list(
        target_instance_score, pred_segments_score, num_instance_pred
    )

    return target_list, pred_list

  @property
  def metrics(self):
    return list(self.metrics_dict.values())

  def eval_metrics(self, target_list, pred_list, starts_with='aggregate'):
    res_dict = {}
    for m in self.metrics_dict:
      if m.startswith(starts_with):
        self.metrics_dict[m].update_state(target_list, pred_list)
        res_dict[m] = self.metrics_dict[m].result()

    return res_dict
