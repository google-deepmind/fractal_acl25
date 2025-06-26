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

"""Process and create datasets for FirA."""
import argparse
import warnings

from nltk.tokenize import sent_tokenize  # pylint: disable=g-importing-member
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # pylint: disable=unused-import


warnings.filterwarnings('ignore')


def load_data(data_folder_path):
  """Load raw data."""
  index_names = {
      'queries.tsv': 'query_id',
      'documents.tsv': 'doc_id',
      'fira-trec19-raw-judgements.tsv': 'id',
  }
  df_dict = {}  # pylint: disable=redefined-outer-name
  for k in index_names:
    df = pd.read_csv(data_folder_path+'/'+k, sep='\t').set_index(
        index_names[k]
    )
    df_dict[k] = df
  df_dict['fira-trec19-raw-judgements.tsv'] = (
      df_dict['fira-trec19-raw-judgements.tsv']
      .reset_index()
      .set_index(['documentId', 'queryId'])
  )
  return df_dict


def process_dataset(df_dict, sentence_score='max'):  # pylint: disable=redefined-outer-name
  """Process and create datasets for FirA."""
  w = 0
  assert sentence_score == 'max'
  # sentence_scores = tf.zeros()
  max_sentences = 250  # pylint: disable=unused-variable
  num_samples = len(df_dict['fira-trec19-raw-judgements.tsv'])   # pylint: disable=unused-variable
  df_processed = pd.DataFrame(
      columns=[
          'doc_id',
          'query_id',
          'doc_text_sentences',
          'query_text',
          'sentences_scores',
          'num_sentences',
      ]
  ).set_index(['doc_id', 'query_id'])

  for doc_id, query_id in df_dict['fira-trec19-raw-judgements.tsv'].index:
    # row_dict = {'doc_id': doc_id, 'query_id': query_id}
    row_dict = {}
    try:
      doc_text = df_dict['documents.tsv'].loc[doc_id, 'doc_text']
      query_text = df_dict['queries.tsv'].loc[query_id, 'query_text']
      num_words = len(doc_text.split(' '))  # pylint: disable=unused-variable
      doc_text_sentences = sent_tokenize(doc_text)
      row_dict['doc_text_sentences'] = doc_text_sentences
      row_dict['query_text'] = query_text
      row_dict['num_sentences'] = len(doc_text_sentences)
      doc_query_pair = (
          df_dict['fira-trec19-raw-judgements.tsv']
          .loc[doc_id, query_id]
          .set_index('id')
          .sort_index()
      )
      doc_sentences_words = [  # pylint: disable=unused-variable
          sentence.split(' ') for sentence in doc_text_sentences
      ]
      doc_word_avg = [
          np.zeros(len(sentence.split(' '))) for sentence in doc_text_sentences
      ]
      doc_text_sentences = [sentence + ' ' for sentence in doc_text_sentences]
      char_len_sentences = [len(sentence) for sentence in doc_text_sentences]
      for idx in doc_query_pair.index:
        relevant_spans = (
            doc_query_pair['relevanceCharacterRanges'].loc[idx].split(';')
        )
        relevance_score = int(
            doc_query_pair['relevanceLevel'].loc[idx].split('_')[0]
        )
        if relevant_spans[0] != '<no ranges selected>':
          for span in relevant_spans:
            start, end = int(span.split('-')[0]), int(span.split('-')[1])
            doc_text_relevant_span = doc_text[start : end + 1]  # pylint: disable=unused-variable
            cumulative_char = 0
            start_sentence_id, end_sentence_id = None, None
            for s in range(len(char_len_sentences)):
              cumulative_char += char_len_sentences[s]
              if start <= cumulative_char and start_sentence_id is None:
                start_sentence_id = s
                start = cumulative_char - char_len_sentences[s]
              if end <= cumulative_char and end_sentence_id is None:
                end_sentence_id = s
                end = cumulative_char - char_len_sentences[s]
            for s in range(start_sentence_id, end_sentence_id + 1):
              start_word_id = 0
              end_word_id = len(doc_word_avg[s]) - 1
              if start_sentence_id == s:
                start_word_id = len(doc_text_sentences[s][:start].split(' '))
              if end_sentence_id == s:
                end_word_id = len(doc_text_sentences[s][end + 1 :].split(' '))
              for sss in range(
                  len(doc_word_avg[s][start_word_id : end_word_id + 1])
              ):
                doc_word_avg[s][sss] = max(
                    relevance_score, doc_word_avg[s][sss]
                )
            break
        else:
          for s in range(len(doc_word_avg)):
            for sss in range(len(doc_word_avg[s])):
              doc_word_avg[s][sss] = max(relevance_score, doc_word_avg[s][sss])
      if sentence_score == 'max':
        doc_word_avg = [max(doc_word_avg[s]) for s in range(len(doc_word_avg))]

      row_dict['sentences_scores'] = doc_word_avg
      df_processed.loc[(doc_id, query_id), :] = row_dict
      w += 1
    except:  # pylint: disable=bare-except
      pass
  return df_processed


# df -> tf dataset
def create_dataset(
    df_processed,
    dataset_path,
    aggregation_type='max',
    encoder_type='sentence-t5',
):
  """Create tf dataset."""
  target_agg_score = None
  target_instance_score = df_processed['sentences_scores']
  if aggregation_type == 'max':
    target_agg_score = [
        max(target_instance_score[i]) for i in range(len(target_instance_score))
    ]
  elif aggregation_type == 'mean':
    target_agg_score = [
        sum(target_instance_score[i]) / len(target_instance_score[i])
        for i in range(len(target_instance_score))
    ]
  query_text = df_processed['query_text'].values
  num_segments = df_processed['num_sentences'].values.tolist()
  doc_text = df_processed['doc_text_sentences'].explode().values
  # Max number of predicted segments
  max_segments = tf.keras.backend.get_value(tf.reduce_max(num_segments))  # pylint: disable=unused-variable
  target_instance_score = [
      tf.constant(target_instance_score[i]) for i in range(len(df_processed))
  ]
  ds = None  # pylint: disable=unused-variable
  if encoder_type == 'sentence-t5':
    sentence_t5_url = 'https://tfhub.dev/google/sentence-t5/st5-large/1'
    sentence_t5_encoder = hub.KerasLayer(sentence_t5_url)
    query_embedding = tf.concat(
        [
            sentence_t5_encoder(query_text[x : x + 500])[0]
            for x in range(0, len(query_text), 500)
        ],
        0,
    )
    doc_embedding = tf.concat(
        [
            sentence_t5_encoder(doc_text[x : x + 500])[0]
            for x in range(0, len(doc_text), 500)
        ],
        0,
    )
    doc_embedding = tf.RaggedTensor.from_row_lengths(
        doc_embedding, num_segments
    )
    target_instance_score = tf.ragged.stack(target_instance_score)
    ds = tf.data.Dataset.from_tensor_slices((
        query_embedding,
        doc_embedding,
        num_segments,
        target_agg_score,
        target_instance_score,
    ))
    ds = ds.shuffle(len(ds))
    ds.save(f'{dataset_path}/{encoder_type}_max_agg_multiclass')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', type=str, help='Path to FirA directory.')
  args = parser.parse_args()
  df_dict = load_data(args.data_dir)
  df_processed_max = process_dataset(df_dict)
  create_dataset(df_processed_max, args.data_dir)
