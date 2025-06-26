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

"""FRACTAL model for FirA retrieval task (multiclass classification)."""
from retrieval.FirA.utils import correlation_prior
from retrieval.FirA.utils import cosine_similarity_prior
from retrieval.FirA.utils import loss_fn
from sklearn.metrics import roc_auc_score  # pylint: disable=g-importing-member
import tensorflow as tf


def get_pseudolabels_loader(
    model, dataloader, aggregation_fn='max', batch_size=32
):
  """Generates a dataloader with pseudolabels for the given prior model.

  Args:
    model: The model to use for generating pseudolabels.
    dataloader: The dataloader to generate pseudolabels for.
    aggregation_fn: The aggregation function.
    batch_size: The batch size to use for the generated dataloader.

  Returns:
    A dataloader with pseudolabels.
  """
  pslab_ds = None
  for batch_id, batch in enumerate(dataloader):
    (
        questions_embedding,
        context_segments_embedding,
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
    for bag_id, num_segments_bag in enumerate(num_context_segments):
      bag_inst_preds = pred_context_segment_scores[
          bag_id, :num_segments_bag, :
      ]  # shape: [num_instances, num_classes]
      bag_agg_target = target_agg_score[bag_id]
      bag_segments = num_context_segments[bag_id]  # pylint: disable=unused-variable
      bag_instances_pseudolabels = tf.cast(
          tf.argmax(bag_inst_preds, axis=-1), tf.float32
      )
      if aggregation_fn == 'max':
        bag_agg_pred = tf.reduce_max(bag_instances_pseudolabels)
        if bag_agg_pred != bag_agg_target:
          if bag_agg_target > bag_agg_pred:
            flip_label_idx = [tf.argmax(bag_inst_preds[:, int(bag_agg_target)])]
            # bag_instances_pseudolabels[flip_label_idx] = bag_agg_target
            bag_instances_pseudolabels = tf.tensor_scatter_nd_update(
                bag_instances_pseudolabels, [flip_label_idx], [bag_agg_target]
            )
          else:
            for inst_id, inst_psl in enumerate(bag_instances_pseudolabels):
              if inst_psl > bag_agg_target:
                flip_label_idx = [inst_id]
                new_label = float(
                    tf.argmax(
                        bag_inst_preds[inst_id, : int(bag_agg_target) + 1]
                    )
                )
                bag_instances_pseudolabels = tf.tensor_scatter_nd_update(
                    bag_instances_pseudolabels, [flip_label_idx], [new_label]
                )
      batch_pseudolabels.append(bag_instances_pseudolabels)
    batch_pseudolabels_ragged = tf.ragged.stack(batch_pseudolabels)
    ds_batch = tf.data.Dataset.from_tensor_slices((
        questions_embedding,
        context_segments_embedding,
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
  """FRACTAL model for FirA retrieval task (multiclass classification)."""

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
        'instance_mae': tf.keras.metrics.MeanAbsoluteError(name='instance_mae'),
        'aggregate_mae': tf.keras.metrics.MeanAbsoluteError(
            name='aggregate_mae'
        ),
        'instance_mse': tf.keras.metrics.MeanSquaredError(name='instance_mse'),
        'aggregate_mse': tf.keras.metrics.MeanSquaredError(
            name='aggregate_mse'
        ),
    }

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
    pred_prob = []
    for i in range(len(num_segments)):
      target += (
          tf.nest.flatten(target_instance_score[i, : num_segments[i]])[0]
          .numpy()
          .tolist()
      )
      pred_prob += pred_segments_score[i, : num_segments[i]].numpy().tolist()
      pred += (
          tf.argmax(pred_segments_score[i, : num_segments[i]], axis=1)
          .numpy()
          .tolist()
      )
    return target, pred, pred_prob

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
      pred.append(
          agg_fn(tf.argmax(pred_segments_score[i, : num_segments[i]], axis=-1))
      )
    return target, pred

  def aggregate_loss(
      self, loss_fn, target_agg_score, pred_segments_score, num_segments  # pylint: disable=redefined-outer-name
  ):
    total_loss = tf.constant(0.0, dtype=tf.float32)

    for bag_id, num_segments_bag in enumerate(num_segments):
      bag_inst_preds = pred_segments_score[
          bag_id, :num_segments_bag, :
      ]  # shape: [num_instances, num_classes]
      num_classes = bag_inst_preds.shape[-1]

      # Initialize as a tensor instead of tf.Variable
      bag_pred_agg_scores = tf.zeros([num_classes], dtype=tf.float32)

      for cid in range(num_classes):
        product_of_sum_prob = tf.constant(1.0, dtype=tf.float32)
        for i in range(bag_inst_preds.shape[0]):
          for j in range(bag_inst_preds.shape[0]):
            if i != j:
              product_of_sum_prob *= tf.reduce_sum(bag_inst_preds[j, :cid])

          update = bag_inst_preds[i, cid] * product_of_sum_prob

          # Accumulate in a tensor way
          update_vec = tf.one_hot(cid, num_classes) * update
          bag_pred_agg_scores += update_vec

        # Divide by number of instances
        bag_pred_agg_scores = tf.tensor_scatter_nd_update(
            bag_pred_agg_scores,
            indices=[[cid]],
            updates=[
                bag_pred_agg_scores[cid]
                / tf.cast(bag_inst_preds.shape[-1], tf.float32)
            ],
        )

        # Add product term (make sure it’s differentiable)
        prod_term = tf.reduce_prod(bag_inst_preds[:, cid])
        bag_pred_agg_scores += tf.one_hot(cid, num_classes) * prod_term

      # Compute loss from model-connected predictions
      total_loss += loss_fn(
          tf.expand_dims(target_agg_score[bag_id], 0),
          tf.expand_dims(bag_pred_agg_scores, 0),
      )

    return total_loss

  def instance_error(
      self, loss_fn, target_instance_score, pred_segments_score, num_segments  # pylint: disable=redefined-outer-name
  ):
    total_loss = tf.constant(0.0, dtype=tf.float32)
    for bag_id, num_segments_bag in enumerate(num_segments):
      bag_inst_preds = pred_segments_score[
          bag_id, :num_segments_bag, :
      ]  # shape: [num_instances, num_classes]
      bag_inst_targets = target_instance_score[
          bag_id, :num_segments_bag
      ]  # shape: [num_instances]
      total_loss = tf.concat(loss_fn(bag_inst_targets, bag_inst_preds), axis=0)

    return total_loss

  def call(self, batch):
    (
        questions_embedding,
        context_segments_embedding,
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
      summary_segment_scores.append(self.nli_mlp(mlp_input))
    pred_context_segment_scores = tf.stack(summary_segment_scores, axis=2)
    pred_context_segment_scores = tf.transpose(
        pred_context_segment_scores, perm=[0, 2, 1]
    )

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
    (
        batch_target_instance_list,
        batch_pred_instance_list,
        _,
    ) = self.get_instance_list(
        target_instance_score, pred_segments_score, num_segments
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
        _,
        _,
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
    (
        batch_target_instance_list,
        batch_pred_instance_list,
        _,
    ) = self.get_instance_list(
        target_instance_score, pred_segments_score, num_segments
    )
    metrics_result = self.eval_metrics(
        batch_target_instance_list,
        batch_pred_instance_list,
        batch_target_agg_list,
        batch_pred_agg_list,
    )

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
            tf.cast(target_instance_list, tf.float32),
            tf.cast(pred_instance_list, tf.float32)
        )
      elif m.startswith('aggregate'):
        self.metrics_dict[m].update_state(target_agg_list, pred_agg_list)

    return {m: self.metrics_dict[m].result() for m in self.metrics_dict.keys()}
