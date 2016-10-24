import tensorflow as tf
import tempfile
import export

# A single serialized example
# (You can read this from a file using TFRecordReader)
ex = export.make_example([1, 2, 3], [0, 1, 0]).SerializeToString()
 
# Define how to parse the example
context_features = {
    "length": tf.FixedLenFeature([], dtype=tf.int64)
}
sequence_features = {
    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
}
 
# Parse the example
context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    serialized=ex,
    context_features=context_features,
    sequence_features=sequence_features
)
