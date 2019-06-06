import pickle
import numpy as np

MAX_SENT_NUM_ROUGH = 20
MAX_SENT_LEN_ROUGH = 150
MAX_DOC_LEN_ROUGH = 500

MAX_SENT_NUM_RIGOUR = 20
MAX_SENT_LEN_RIGOUR = 100
MAX_DOC_LEN_RIGOUR = 400

NUM_CLASS = 10

def read_data(data_path):
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)
    return data[0], data[1]


def stuff_doc(doc, model_option, data_option):
    '''
    :param doc:
    :param model_option: 0-Han, 1-TextCnn
    :param data_option: 0-rough, 1-rigour
    :return:
    '''
    def stuff_sent(sentence, max_length):
        if len(sentence) > max_length:
            start_index = (len(sentence) - max_length - 1) // 2 + 1
            sentence = sentence[start_index: start_index + max_length]
        else:
            sentence.extend([0]*(max_length-len(sentence)))
        return sentence

    if model_option == 0:
        if data_option == 0:
            max_sentence_num = MAX_SENT_NUM_ROUGH
            max_sentence_length = MAX_SENT_LEN_ROUGH
        elif data_option == 1:
            max_sentence_num = MAX_SENT_NUM_RIGOUR
            max_sentence_length = MAX_SENT_LEN_RIGOUR
        else:
            raise ValueError('Parameter error! : "data_option" must be 0 or 1')

        if len(doc) > max_sentence_num:
            start_index = (len(doc)-max_sentence_num-1)//2+1
            doc = doc[start_index : start_index + max_sentence_num]

        doc = np.array(list(map(lambda sent: stuff_sent(sent, max_sentence_length), doc)))
        if len(doc) < max_sentence_num:
            doc = np.pad(doc, ((0, max_sentence_num - len(doc)), (0, 0)), 'constant')
        return doc
    elif model_option == 1:
        if data_option == 0:
            max_doc_length = MAX_DOC_LEN_ROUGH
        elif data_option == 1:
            max_doc_length = MAX_DOC_LEN_RIGOUR
        else:
            raise ValueError('Parameter error! : "data_option" must be 0 or 1')
        doc = stuff_sent(doc, max_doc_length)
        return doc
    else:
        raise ValueError('Parameter error! : "model_option" must be 0 or 1')

def gen_one_hot(label):
    label = np.array(label)
    return (np.arange(NUM_CLASS)==label[:,None]).astype(np.int32)

#Han
# def decode_from_tfrecord(filename_queue):
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'input_raw': tf.FixedLenFeature([], tf.int64),
#                                        })  # 取出包含input_raw和label的feature对象
#     input_raw = tf.reshape(features['input_raw'], [max_sentence_num, max_sentence_length])
#     label = features['label']
#     return input_raw, label
#
#
# def batch_reader(filenames, batch_size, thread_count, num_epochs=None):
#     filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
#     input_raw, label = decode_from_tfrecord(filename_queue)
#     min_after_dequeue = 1000
#     capacity = min_after_dequeue + thread_count * batch_size
#     x_batch, y_batch = tf.train.shuffle_batch([input_raw, label],
#                                               batch_size=batch_size,
#                                               capacity=capacity,
#                                               min_after_dequeue=min_after_dequeue,
#                                               num_threads=thread_count)
#     y_batch = (np.arange(class_num) == y_batch[:, None]).astype(np.int32)
#
#     return x_batch, y_batch

#TextCnn
# def decode_from_tfrecord(filename_queue):
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'input_raw': tf.FixedLenFeature([], tf.int64),
#                                        })  # 取出包含input_raw和label的feature对象
#     input_raw = features['input_raw']
#     label = features['label']
#     return input_raw, label
#
#
# def batch_reader(filenames, batch_size, thread_count, num_epochs=None):
#     filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
#     input_raw, label = decode_from_tfrecord(filename_queue)
#     min_after_dequeue = 1000
#     capacity = min_after_dequeue + thread_count * batch_size
#     x_batch, y_batch = tf.train.shuffle_batch([input_raw, label],
#                                               batch_size=batch_size,
#                                               capacity=capacity,
#                                               min_after_dequeue=min_after_dequeue,
#                                               num_threads=thread_count)
#     y_batch = (np.arange(class_num) == y_batch[:, None]).astype(tf.int32)
#
#     return x_batch, y_batch