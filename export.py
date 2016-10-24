import tensorflow as tf
import tempfile

sequences = [[1, 2, 3], [4, 5, 1], [1, 2]]
label_sequences = [[0, 1, 0], [1, 0, 0], [1, 1]]

def make_example(sequences, labels):
	ex = tf.train.SequenceExample()
	# non-sequential features
	sequence_length = len(sequences)
	ex.context.feature['length'].int64_list.value.append(sequence_length)
	fl_tokens = ex.feature_lists.feature_list['tokens']
	fl_labels = ex.feature_lists.feature_list['labels']

	for token, label in zip(sequences, labels):
		fl_tokens.feature.add().int64_list.value.append(token)
		fl_labels.feature.add().int64_list.value.append(label)
	return ex


with tempfile.NamedTemporaryFile() as fp:
	writer = tf.python_io.TFRecordWriter(fp.name)
	for sequence, label_sequence in zip(sequences, label_sequences):
		ex = make_example(sequence, label_sequence)
		writer.write(ex.SerializeToString())
	writer.close()

