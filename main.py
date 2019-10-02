from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np

print("Tensorflow ver.- ", tf.__version__)
print("TPandas ver.- ", pd.__version__)

FEATURES = ['sizeH', 'sizeV', 'position', 'toUp', 'toDown', 'toLeft', 'toRight']
RESULT = ['up', 'down', 'left', 'right']

# position = dict({'position': np.random.rand(1, 9)})
# print(position)


def input_evaluation_set():
    features = {
        'sizeH': [3, ],
        'sizeV': [3, ],
        'position': [[1, 2, 3, 4, 5, 6, 7, 0, 8], ],
        'toUp': [[1, 2, 3, 4, 5, 6, 7, 0, 8, ], ],
        'toDown': [[1, 2, 3, 4, 5, 6, 7, -1, 8], ],
        'toLeft': [[1, 2, 3, 4, 5, 6, 0, 7, 8], ],
        'toRight': [[1, 2, 3, 4, 5, 6, 7, 8, 0], ],
    }
    labels = pd.Series([RESULT.index('right'), ])

    return features, labels


def input_fn(features, labels, batch_size=256):
    """An input function for training or evaluating"""
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.batch(batch_size)


def input_prediction_fn(features, batch_size=256):
    #An input function for prediction.
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

# train_path = tf.keras.utils.get_file(
#  "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")

# test_path = tf.keras.utils.get_file(
#    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

# train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
# test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# train_y = train.pop('Species')
# test_y = test.pop('Species')

#print(train.head())
my_feature_columns = []
# my_feature_columns = [(tf.compat.v2.feature_column.numeric_column(key=FEATURES[0]))]
# my_feature_columns.append(tf.compat.v2.feature_column.numeric_column(key=FEATURES[0]))

for int_key in FEATURES[0:2]:
    my_feature_columns.append(tf.feature_column.numeric_column(key=int_key))
for emb_key in FEATURES[2:]:
    my_feature_columns.append(tf.feature_column.numeric_column(key=emb_key, shape=[1, 9]))


classifier = tf.compat.v2.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[48, 8],
    # The model must choose between 3 classes.
    n_classes=4,
    model_dir='./output')

train, train_y = input_evaluation_set()

classifier.train(
    input_fn=lambda: input_fn(train, train_y),
    steps=10)
# latest_checkpoint = classifier.latest_checkpoint
# print(latest_checkpoint)
variable_names = classifier.get_variable_names()
for name in variable_names:
    print(name, classifier.get_variable_value(name))

predict_x = {
    'sizeH': [3, 3],
    'sizeV': [3, 3],
    'position': [[1, 2, 3, 4, 5, 6, 7, 0, 8], [1, 2, 3, 4, 5, 6, 0, 7, 8]],
    'toUp': [[1, 2, 3, 4, 5, 6, 7, 0, 8, ], [1, 2, 3, 0, 5, 6, 4, 7, 8]],
    'toDown': [[1, 2, 3, 4, 5, 6, 7, -1, 8], [1, 2, 3, 4, 5, 6, -1, 7, 8]],
    'toLeft': [[1, 2, 3, 4, 5, 6, 0, 7, 8], [1, 2, 3, 4, 5, 6, -1, 7, 8]],
    'toRight': [[1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 0, 8]],
}

predictions = classifier.predict(
    input_fn=lambda: input_prediction_fn(predict_x))

for pred_dict, expec in zip(predictions, RESULT):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        RESULT[class_id], 100 * probability, expec))


"""{


eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Generate predictions from the model
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}


def input_fn(features, batch_size=256):
    #An input function for prediction.
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        SPECIES[class_id], 100 * probability, expec))

"""
