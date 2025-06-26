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

"""Process the PRM800K dataset and save it as a TF dataset."""
import argparse

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # pylint: disable=unused-import


def process_data(df, encoder, dataset_path, ds_concat=False):  # pylint: disable=redefined-outer-name
  """Process the PRM800K dataset and save it as a TF dataset."""
  for split in df.keys():
    questions = []
    num_segments = []
    solution_steps = []
    rating_steps = []
    first_wrong_step = []
    for i, problem in enumerate(df[split].index):
      question_flag = True
      first_step_flag = True
      num_segments_flag = True
      steps = df[split]['label'][i]['steps']
      for s, step in enumerate(steps):
        if (
            step['completions'] is not None
            and step['completions'][0] is not None
            and step['completions'][0]['text'] is not None
            and step['completions'][0]['rating'] is not None
        ):
          solution_steps.append(step['completions'][0]['text'])
          rating_steps.append(int(step['completions'][0]['rating']))
          if first_step_flag:
            first_wrong_step.append(-1)
            first_step_flag = False
          if question_flag:
            questions.append(problem['problem'])
            question_flag = False
          if not num_segments_flag:
            num_segments[-1] += 1
          else:
            num_segments.append(1)
            num_segments_flag = False
          if (
              first_wrong_step[-1] == -1
              and step['completions'][0]['rating'] == -1
          ):
            first_wrong_step[-1] = s

      assert sum(num_segments) == len(solution_steps)
    target_instance_labels = tf.RaggedTensor.from_row_lengths(
        rating_steps, num_segments
    )
    solution_steps_embedding = tf.concat(
        [
            encoder(tf.constant(solution_steps[x : x + 800]))[0]
            for x in range(0, len(solution_steps), 800)
        ],
        axis=0,
    )
    questions_embedding = tf.concat(
        [
            encoder(tf.constant(questions[x : x + 800]))[0]
            for x in range(0, len(questions), 800)
        ],
        axis=0,
    )
    target_aggregate_labels = [1 if f == -1 else 0 for f in first_wrong_step]
    solution_steps_embedding = tf.RaggedTensor.from_row_lengths(
        solution_steps_embedding, num_segments
    )

    ds = tf.data.Dataset.from_tensor_slices((
        solution_steps_embedding,
        questions_embedding,
        num_segments,
        target_aggregate_labels,
        target_instance_labels,
        first_wrong_step,
    ))
    ds = ds.shuffle(len(ds))
    if ds_concat:
      existing_ds = tf.data.Dataset.load(f'{dataset_path}/sentence-t5_{split}')
      ds = ds.concatenate(existing_ds)
      ds = ds.shuffle(len(ds))
    ds.save(f'{dataset_path}/sentence-t5_{split}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'data_dir', type=str, help='Path to PRM800K directory.'
  )
  args = parser.parse_args()
  raw_data = args.data_dir
  # df_phase1_test = pd.read_json(
  #     raw_data + 'phase1_test.jsonl', lines=True
  # ).set_index(
  #     'question'
  # )  # 106 samples
  # df_phase1_train = pd.read_json(
  #     raw_data + 'phase1_train.jsonl', lines=True
  # ).set_index(
  #     'question'
  # )  # 2767 samples
  # df_phase2_test = pd.read_json(
  #     raw_data + 'phase2_test.jsonl', lines=True
  # ).set_index(
  #     'question'
  # )  # 949 samples
  # df_phase2_train = pd.read_json(
  #     raw_data + 'phase2_train.jsonl', lines=True
  # ).set_index(
  #     'question'
  # )  # 97782 samples
  with open(raw_data + '/phase1_test.jsonl', 'r') as f:
    df_phase1_test = pd.read_json(f, lines=True).set_index('question')
  with open(raw_data + '/phase1_train.jsonl', 'r') as f:
    df_phase1_train = pd.read_json(f, lines=True).set_index('question')
  with open(raw_data + '/phase2_test.jsonl', 'r') as f:
    df_phase2_test = pd.read_json(f, lines=True).set_index('question')
  with open(raw_data + '/phase2_train.jsonl', 'r') as f:
    df_phase2_train = pd.read_json(f, lines=True).set_index('question')

  sentence_t5_url = 'https://tfhub.dev/google/sentence-t5/st5-large/1'
  encoder = hub.KerasLayer(sentence_t5_url)
  df = {'test': df_phase1_test, 'train': df_phase1_train}
  process_data(df, encoder, raw_data)
  df = {'test': df_phase2_test, 'train': df_phase2_train}
  process_data(df, encoder, raw_data, ds_concat=True)
