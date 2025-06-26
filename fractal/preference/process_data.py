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

"""Create tf.Dataset for training."""

import argparse
from itertools import islice  # pylint: disable=g-importing-member
import json
from operator import itemgetter  # pylint: disable=g-importing-member
import nltk.data
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # pylint: disable=unused-import


def process_dataset(
    dataset_path, encoder_type='sentence-t5',
    aggregation_type='avg'  # pylint: disable=unused-argument
):
  """Process the dataset."""

  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  # print '\n-----\n'.join(tokenizer.tokenize(data))

  encoder = None
  if encoder_type == 'sentence-t5':
    sentence_t5_url = 'https://tfhub.dev/google/sentence-t5/st5-large/1'
    encoder = hub.KerasLayer(sentence_t5_url)
  json_filenames = {
      'train': f'{dataset_path}/train_feedback.json',
      'dev': f'{dataset_path}/dev_feedback.json',
  }
  for filename in json_filenames:
    with open(json_filenames[filename], 'r') as f:
      raw_dataset = np.array(json.load(f))
    # Format: dataset_pref = (question, passages, pred 1, pred 2, target)
    # Format: dataset_fine = (question, passages, pred, fine annotations)
    questions_fine, passages_fine, pred_fine, target_fine = [], [], [], []
    questions_pref, passages_pref, pred1_pref, pred2_pref, target_pref = (
        [],
        [],
        [],
        [],
        [],
    )
    num_passages_per_sample = []
    num_instance_pred1_pref, num_instance_pred2_pref = [], []
    num_instance_pred_fine = []
    num_passage_splits = []
    num_pairs_per_sample = []
    pred_pair_names = {
        0: {'pred1': 'prediction 1', 'pred2': 'prediction 2'},
        1: {'pred1': 'prediction 1', 'pred2': 'prediction 3'},
        2: {'pred1': 'prediction 1', 'pred2': 'prediction 4'},
        3: {'pred1': 'prediction 2', 'pred2': 'prediction 3'},
        4: {'pred1': 'prediction 2', 'pred2': 'prediction 4'},
        5: {'pred1': 'prediction 3', 'pred2': 'prediction 4'},
    }
    for sample_dict in raw_dataset:
      num_passages_per_sample.append(len(sample_dict['passages']))
      passages = [' '.join(p) for p in sample_dict['passages']]
      combined_passage = (' '.join(passages)).split()
      passage_split_sample = [
          ' '.join(combined_passage[x : x + 500])
          for x in range(0, len(combined_passage), 500)
      ]
      num_passage_splits.append(len(passage_split_sample))
      # dataset_pref
      sample_pref = sample_dict['preference']
      num_pairs = 0
      for pair_id in range(len(sample_pref)):
        num_pairs += 1
        pred1_pref_sentence_split = tokenizer.tokenize(
            sample_dict[pred_pair_names[pair_id]['pred1']]
        )
        num_instance_pred1_pref.append(len(pred1_pref_sentence_split))
        pred2_pref_sentence_split = tokenizer.tokenize(
            sample_dict[pred_pair_names[pair_id]['pred2']]
        )
        num_instance_pred2_pref.append(len(pred2_pref_sentence_split))
        pred1_pref += pred1_pref_sentence_split
        pred2_pref += pred2_pref_sentence_split
        target_pref.append(sample_pref[pair_id])
      num_pairs_per_sample.append(num_pairs)

      # dataset_fine
      questions_fine.append(sample_dict['question'])
      passages_fine += passage_split_sample
      pred_sentence_split = tokenizer.tokenize(sample_dict['prediction 1'])
      num_instance_pred_fine.append(len(pred_sentence_split))
      pred_fine += pred_sentence_split
      sample_fine_feedback = sample_dict['feedback']
      sample_errors = sample_fine_feedback['errors']
      # pred_word_split = sample_dict['prediction 1'].split()
      num_words_in_pred_sentence = [
          len(sent.split()) for sent in pred_sentence_split
      ]
      target_word_labels = []
      irrelevant_spans = []
      for error in sample_errors:
        if error['error type'] == 'Irrelevant':
          irrelevant_spans.append((error['start'], error['end']))
      irrelevant_span_sorted = sorted(irrelevant_spans, key=itemgetter(0))

      if len(irrelevant_span_sorted) == 0:  # pylint: disable=g-explicit-length-test
        target_fine.append([1.0] * len(pred_sentence_split))
      else:
        for i, (start_index, end_index) in enumerate(irrelevant_span_sorted):
          if i == 0 and start_index > 0:
            target_word_labels += [1.0] * len(
                sample_dict['prediction 1'][:start_index].split()
            )
          elif i > 0 and irrelevant_span_sorted[i - 1][1] < start_index:
            target_word_labels += [1.0] * len(
                sample_dict['prediction 1'][
                    irrelevant_span_sorted[i - 1][1] + 1 : start_index
                ].split()
            )
          elif i > 0 and irrelevant_span_sorted[i - 1][1] >= start_index:
            start_index = irrelevant_span_sorted[i - 1][1] + 1
          if start_index > 0:
            if (
                sample_dict['prediction 1'][start_index - 1] != ' '
                and sample_dict['prediction 1'][start_index] != ' '
            ):
              span_split = sample_dict['prediction 1'][
                  start_index : end_index + 1
              ].split()
              start_index = len(span_split[0]) + 1
          target_word_labels += [0.0] * len(
              sample_dict['prediction 1'][start_index : end_index + 1].split()
          )
        target_word_labels = [
            list(islice(target_word_labels, elem))
            for elem in num_words_in_pred_sentence
        ]
        assert len(target_word_labels) == len(pred_sentence_split)
        target_fine.append(
            [min(word_labels_list) for word_labels_list in target_word_labels]
        )

    passages_splits_fine_embedding = tf.concat(
        [
            encoder(tf.constant(passages_fine[x : x + 500]))[0]
            for x in range(0, len(passages_fine), 500)
        ],
        axis=0,
    )
    questions_fine_embedding = tf.concat(
        [
            encoder(tf.constant(questions_fine[x : x + 500]))[0]
            for x in range(0, len(questions_fine), 500)
        ],
        axis=0,
    )
    pred_fine_embedding = tf.concat(
        [
            encoder(tf.constant(pred_fine[x : x + 500]))[0]
            for x in range(0, len(pred_fine), 500)
        ],
        axis=0,
    )
    pred1_pref_embedding = tf.concat(
        [
            encoder(tf.constant(pred1_pref[x : x + 500]))[0]
            for x in range(0, len(pred1_pref), 500)
        ],
        axis=0,
    )
    pred2_pref_embedding = tf.concat(
        [
            encoder(tf.constant(pred2_pref[x : x + 500]))[0]
            for x in range(0, len(pred2_pref), 500)
        ],
        axis=0,
    )
    passages_splits_fine_embedding = tf.RaggedTensor.from_row_lengths(
        passages_splits_fine_embedding, num_passage_splits
    )
    pred_fine_embedding = tf.RaggedTensor.from_row_lengths(
        pred_fine_embedding, num_instance_pred_fine
    )
    pred1_pref_embedding = tf.RaggedTensor.from_row_lengths(
        pred1_pref_embedding, num_instance_pred1_pref
    )
    pred2_pref_embedding = tf.RaggedTensor.from_row_lengths(
        pred2_pref_embedding, num_instance_pred2_pref
    )

    for i, passage_embedding in enumerate(passages_splits_fine_embedding):
      passages_pref += list(
          tf.repeat([passage_embedding], [num_pairs_per_sample[i]], axis=0)
      )
      questions_pref += list(
          tf.repeat(
              [questions_fine_embedding[i]], [num_pairs_per_sample[i]], axis=0
          )
      )
    num_passage_splits_pref = tf.repeat(
        tf.constant(num_passage_splits), repeats=num_pairs_per_sample
    )
    passages_pref_embedding = tf.ragged.stack(passages_pref)
    questions_pref_embedding = tf.stack(questions_pref)
    target_instance_labels = tf.ragged.stack(target_fine)
    target_aggregate_labels = [
        sum(sent_label) / len(sent_label) for sent_label in target_fine
    ]
    # repeat num_passages
    ds_pref = tf.data.Dataset.from_tensor_slices((
        passages_pref_embedding,
        questions_pref_embedding,
        pred1_pref_embedding,
        pred2_pref_embedding,
        num_passage_splits_pref,
        num_instance_pred1_pref,
        num_instance_pred2_pref,
        target_pref,
    ))
    ds_pref = ds_pref.shuffle(len(ds_pref))
    ds_pref.save(f'{dataset_path}/{encoder_type}_{filename}_preference')
    ds_fine = tf.data.Dataset.from_tensor_slices((
        passages_splits_fine_embedding,
        questions_fine_embedding,
        pred_fine_embedding,
        num_passage_splits,
        num_instance_pred_fine,
        target_aggregate_labels,
        target_instance_labels,
    ))
    ds_fine.save(f'{dataset_path}/{encoder_type}_{filename}_finegrained')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'data_dir', type=str, help='Path to QA feedback directory.'
  )
  args = parser.parse_args()
  process_dataset(args.data_dir)
