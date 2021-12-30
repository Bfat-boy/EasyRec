# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.input.input import Input

if tf.__version__ >= '2.0':
  ignore_errors = tf.data.experimental.ignore_errors()
  tf = tf.compat.v1
else:
  ignore_errors = tf.contrib.data.ignore_errors()


class CSVInput(Input):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1):
    super(CSVInput, self).__init__(data_config, feature_config, input_path,
                                   task_index, task_num)

  def _parse_csv(self, line):
    record_defaults = [
        self.get_type_defaults(t, v)
        for t, v in zip(self._input_field_types, self._input_field_defaults)
    ]

    def _check_data(line):
      sep = self._data_config.separator
      if type(sep) != type(str):
        sep = sep.encode('utf-8')
      field_num = len(line[0].split(sep))
      assert field_num == len(record_defaults), \
          'sep[%s] maybe invalid: field_num=%d, required_num=%d' % \
          (sep, field_num, len(record_defaults))
      return True

    check_op = tf.py_func(_check_data, [line], Tout=tf.bool)
    with tf.control_dependencies([check_op]):
      fields = tf.decode_csv(
          line,
          field_delim=self._data_config.separator,
          record_defaults=record_defaults,
          name='decode_csv')

    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}

    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]
    return inputs

  def _build(self, mode, params):
    file_paths = []
    for x in self._input_path.split(','):
      file_paths.extend(tf.gfile.Glob(x))
    assert len(file_paths) > 0, 'match no files with %s' % self._input_path

    num_parallel_calls = self._data_config.num_parallel_calls
    if mode == tf.estimator.ModeKeys.TRAIN:
      logging.info('train files[%d]: %s' %
                   (len(file_paths), ','.join(file_paths)))
      dataset = tf.data.Dataset.from_tensor_slices(file_paths)
      print('===> sess.run(dataset.make_one_shot_iterator().get_next())=' + 'b../github/EasyRec/data/test/dwd_avazu_ctr_deepmodel_10w.csv')

      if self._data_config.shuffle:
        # shuffle input files
        dataset = dataset.shuffle(len(file_paths))
      # too many readers read the same file will cause performance issues
      # as the same data will be read multiple times
      parallel_num = min(num_parallel_calls, len(file_paths))
      dataset = dataset.interleave(
          tf.data.TextLineDataset,
          cycle_length=parallel_num,
          num_parallel_calls=parallel_num)

      if self._data_config.chief_redundant:
        dataset = dataset.shard(
            max(self._task_num - 1, 1), max(self._task_index - 1, 0))
      else:
        dataset = dataset.shard(self._task_num, self._task_index)
      if self._data_config.shuffle:
        dataset = dataset.shuffle(
            self._data_config.shuffle_buffer_size,
            seed=2020,
            reshuffle_each_iteration=True)
      print('===> self.num_epochs=' + str(self.num_epochs))
      dataset = dataset.repeat(self.num_epochs)
    else:
      logging.info('eval files[%d]: %s' %
                   (len(file_paths), ','.join(file_paths)))
      dataset = tf.data.TextLineDataset(file_paths)
      dataset = dataset.repeat(1)

    dataset = dataset.batch(self._data_config.batch_size)
    dataset = dataset.map(
        self._parse_csv, num_parallel_calls=num_parallel_calls)
    
    print(""" ===> 此时的dataset为:
           {
             'hour': array([b'10', b'10', b'10'], dtype=object),
             'c1': array([b'1005', b'1005', b'1002'], dtype=object),
             'banner_pos': array([b'0', b'0', b'0'], dtype=object),
             'site_id': array([b'85f751fd', b'4bf5bbe2', b'b15e894b'], dtype=object),
             'site_domain': array([b'c4e18dd6', b'6b560cc1', b'c4e18dd6'], dtype=object),
             'site_category': array([b'50e219e0', b'28905ebd', b'50e219e0'], dtype=object),
             'app_id': array([b'0e8e4642', b'ecad2386', b'ecad2386'], dtype=object),
             'app_domain': array([b'b408d42a', b'7801e8d9', b'7801e8d9'], dtype=object),
             'app_category': array([b'09481d60', b'07d7df22', b'07d7df22'], dtype=object),
             'device_id': array([b'a99f214a', b'a99f214a', b'25a0da4e'], dtype=object),
             'device_ip': array([b'5deb445a', b'447d4613', b'1783ac3d'], dtype=object),
             'device_model': array([b'f4fffcd0', b'cdf6ea96', b'12edfe21'], dtype=object),
             'device_type': array([b'1', b'1', b'0'], dtype=object),
             'device_conn_type': array([b'0', b'0', b'0'], dtype=object),
             'c14': array([b'2098', b'2373', b'2399'], dtype=object),
             'c15': array([b'32', b'32', b'32'], dtype=object),
             'c16': array([b'5', b'5', b'5'], dtype=object),
             'c17': array([b'238', b'272', b'275'], dtype=object),
             'c18': array([b'0', b'3', b'0'], dtype=object),
             'c19': array([b'56', b'5', b'4'], dtype=object),
             'c20': array([b'0', b'0', b'0'], dtype=object),
             'c21': array([b'5', b'3', b'8'], dtype=object),
             'click': array([1, 1, 0])
         }
    """)
    if self._data_config.ignore_error:
      dataset = dataset.apply(ignore_errors)
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    dataset = dataset.map(
        map_func=self._preprocess, num_parallel_calls=num_parallel_calls)

    dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset
