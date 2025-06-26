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

"""Process MultiSpanQA dataset."""
import argparse
from itertools import islice  # pylint: disable=g-importing-member
import json

import nltk
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # pylint: disable=unused-import


def process_dataset(
    dataset_path, encoder_type='sentence-t5', aggregation_type='avg'  # pylint: disable=unused-argument
):
  """Process MultiSpanQA dataset."""

  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  encoder = None
  if encoder_type == 'sentence-t5':
    sentence_t5_url = 'https://tfhub.dev/google/sentence-t5/st5-large/1'
    encoder = hub.KerasLayer(sentence_t5_url)

  json_filenames = {
      'train': f'{dataset_path}/train.json',
      'valid': f'{dataset_path}/valid.json',
  }
  for filename in json_filenames:
    with open(json_filenames[filename], 'r') as f:
      raw_dataset = np.array(json.load(f)['data'])

    questions, context_sentence_splits, num_context_sentences = [], [], []
    target_aggregate_labels, target_instance_labels = [], []
    sample_idx = []

    negative_sample_idx = sorted(
        np.random.choice(np.arange(len(raw_dataset)), len(raw_dataset) // 2)
    )

    for idx, sample_dict in enumerate(raw_dataset):
      sample_idx.append(sample_dict['id'])
      question_word_split = sample_dict['question']
      context_word_labels = sample_dict['label']
      questions.append(' '.join(question_word_split))
      context_sentences = tokenizer.tokenize(' '.join(sample_dict['context']))
      context_word_split = [sent.split() for sent in context_sentences]
      num_words_in_context_sentences = [
          len(sent) for sent in context_word_split
      ]
      assert sum(num_words_in_context_sentences) == len(context_word_labels)
      context_word_labels = [
          list(islice(context_word_labels, elem))
          for elem in num_words_in_context_sentences
      ]

      target_context_label = []
      for _, sentence_word_label in enumerate(context_word_labels):
        unique_label = np.unique(sentence_word_label)
        if 'B' in unique_label or 'I' in unique_label:
          target_context_label.append(1.0)
        else:
          target_context_label.append(0.0)
      if idx in negative_sample_idx:
        pos_sentences_idx = np.where(np.array(target_context_label) == 1.0)[0]
        if len(pos_sentences_idx) != 0 and len(pos_sentences_idx) != len(  # pylint: disable=g-explicit-length-test
            context_sentences
        ):
          diff_sentences = []
          # if 0 not in pos_sentences_idx:
          diff_sentences.append(pos_sentences_idx[0])
          diff_sentences += [
              pos_sentences_idx[s + 1] - pos_sentences_idx[s] - 1
              for s in range(len(pos_sentences_idx) - 1)
          ]
          # if len(context_sentences) - 1 not in pos_sentences_idx:
          diff_sentences.append(
              len(context_sentences) - 1 - pos_sentences_idx[-1]
          )
          neg_chuck_id = np.argmax(diff_sentences)
          if neg_chuck_id == 0:
            context_sentences = context_sentences[: pos_sentences_idx[0]]
          elif neg_chuck_id == len(diff_sentences) - 1:
            context_sentences = context_sentences[pos_sentences_idx[-1] :]
          else:
            context_sentences = context_sentences[
                pos_sentences_idx[neg_chuck_id - 1] : pos_sentences_idx[
                    neg_chuck_id
                ]
            ]
          target_context_label = [0.0] * len(context_sentences)
      target_aggregate_labels.append(max(target_context_label))
      target_instance_labels.append(target_context_label)
      num_context_sentences.append(len(context_sentences))
      context_sentence_splits += context_sentences
    context_splits_embedding = tf.concat(
        [
            encoder(tf.constant(context_sentence_splits[x : x + 500]))[0]
            for x in range(0, len(context_sentence_splits), 500)
        ],
        axis=0,
    )
    questions_embedding = tf.concat(
        [
            encoder(tf.constant(questions[x : x + 500]))[0]
            for x in range(0, len(questions), 500)
        ],
        axis=0,
    )
    target_instance_labels = tf.ragged.stack(target_instance_labels)
    context_splits_embedding = tf.RaggedTensor.from_row_lengths(
        context_splits_embedding, num_context_sentences
    )

    ds = tf.data.Dataset.from_tensor_slices((
        context_splits_embedding,
        questions_embedding,
        num_context_sentences,
        target_aggregate_labels,
        target_instance_labels,
    ))
    ds = ds.shuffle(len(ds))
    ds.save(f'{dataset_path}/{encoder_type}_{filename}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'data_dir',
      type=str,
      help='Path to MultiSpanQA directory: /data/multispan_qa/.',
  )
  args = parser.parse_args()
  df_dict = process_dataset(args.data_dir)
