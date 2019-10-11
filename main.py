from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np
import ast
import puzzlelib

print("Tensorflow ver.- ", tf.__version__)
print("TPandas ver.- ", pd.__version__)

SIZE_H = 3
SIZE_V = 3

DO_NOT_TRAINING = True

#FEATURES = ['size_h', 'size_v', 'position', 'toUp', 'toDown', 'toLeft', 'toRight']
FEATURES = ['size_h', 'size_v', 'position',]
RESULT = ['up', 'down', 'left', 'right']

TRAIN_FILE = './train.csv'
TEST_FILE = './evaluation.csv'

# position = dict({'position': np.random.rand(1, 9)})
# print(position)


def input_from_set(size_h, size_v, position, labels):
    features = {
        'size_h': [size_h, ],
        'size_v': [size_v, ],
        'position': [position, ],
    }
    labels = [labels,]

    return features, labels


def input_from_file(file):

    df = pd.read_csv(file)

    # print(df.head())
    # print(df.dtypes)
    df['position'] = df['position'].apply(lambda s: ast.literal_eval(s))
    df['position'] = df['position'].apply(lambda s: [float(x/10) for x in s])
    # df['toUp'] = df['toUp'].apply(lambda s: ast.literal_eval(s))
    # df['toDown'] = df['toDown'].apply(lambda s: ast.literal_eval(s))
    # df['toLeft'] = df['toLeft'].apply(lambda s: ast.literal_eval(s))
    # df['toRight'] = df['toRight'].apply(lambda s: ast.literal_eval(s))

    labels = df.pop('result')
    return df.to_dict('list'), labels


def input_fn(features, labels, training=True, batch_size=2048):
    """An input function for training or evaluating"""
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1).repeat()
    return dataset.batch(batch_size)


def input_prediction_fn(features, batch_size=1):
    #An input function for prediction.
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

my_feature_columns = []

for int_key in FEATURES[0:2]:
    my_feature_columns.append(tf.feature_column.numeric_column(key=int_key, shape=[1,]))
for emb_key in FEATURES[2:]:
    my_feature_columns.append(tf.feature_column.numeric_column(key=emb_key, shape=[1, 9]))


classifier = tf.compat.v2.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[40, 4,],
    n_classes=4,
    model_dir='./output')
# train, train_y = input_from_set()

test, test_y = input_from_file(TEST_FILE)

if not DO_NOT_TRAINING:
    train, train_y = input_from_file(TRAIN_FILE)
    classifier.train(
        input_fn=lambda: input_fn(train, train_y),
        steps=100000)

eval_result = classifier.evaluate(
                input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# latest_checkpoint = classifier.latest_checkpoint
# print(latest_checkpoint)
# variable_names = classifier.get_variable_names()
# for name in variable_names:
#    print(name, classifier.get_variable_value(name))


pz = puzzlelib.Puzzle(SIZE_H, SIZE_V)
pz.generate()
position = pz.puzzle
maps = []

while True:
    predict_result = 'None'
    maps.append(position)
    all_position = pz.search_all_sets(position)
    if len(all_position) != 4:
        print("FAIL GET ALL POSITION!!!!")
        predict_result = 'FAIL GET ALL POSITION'
        break
    # predict = {'size_h': [SIZE_H,], 'size_v':[SIZE_V,], 'position': [position,], 'toUp': [all_position[0],], 'toDown': [all_position[1],],
    # 'toLeft': [all_position[2],], 'toRight': [all_position[2],]}
    predict = {'size_h': [SIZE_H,], 'size_v':[SIZE_V,], 'position': [position,],}
    predictions = classifier.predict(
        input_fn=lambda: input_prediction_fn(predict))

    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

    print("From position - %s" % position)
    position = all_position[class_id]
    if position == pz.goal:
        print("FINISH: %s" % position)
        predict_result = 'GOOD'
        break
    print("To   position - %s, probality - %s " % (position, probability))
    if len(maps) > 2 and position == maps[-2]:
        print("FAIL!!!! infinity!")
        predict_result = 'TO TRAIN INFINITY'
    if -1 in position:
        print("FAIL!!!!")
        predict_result = 'TO TRAIN'

    if predict_result == 'TO TRAIN':
        pz.set_puzzle(maps[-1])
        searching_result = pz.search_solution()
        searching_result.reverse()
        for j, step in enumerate(searching_result[:(len(searching_result) - 1)]):
            pos = step.set.position
            all_sets = pz.search_all_sets(pos)
            set = searching_result[j+1].set.position
            if set == all_sets[0]:
                classes = 0
            elif set == all_sets[1]:
                classes = 1
            elif set == all_sets[2]:
                classes = 2
            elif set == all_sets[3]:
                classes = 3
            else:
                err = "ERROR to APPLY Classes in search result %s %s " % (set, all_sets)
                raise err
            print('\nTraining to - %s \n' % all_sets[classes])
            train, train_y = input_from_set(SIZE_H, SIZE_V, maps[-1], classes)
            classifier.train(
                input_fn=lambda: input_fn(train, train_y),
                steps=100)
            test, test_y = input_from_set(SIZE_H, SIZE_V, maps[-1], classes)
            eval_result = classifier.evaluate(
                input_fn=lambda: input_fn(test, test_y, training=False))
            print('\nTest set accuracy after training: {accuracy:0.3f}\n'.format(**eval_result))
            position = maps[-1]
            maps.pop()
            break
    elif predict_result == 'TO TRAIN INFINITY':
        pz.set_puzzle(maps[-2])
        searching_result = pz.search_solution()
        searching_result.reverse()
        for j, step in enumerate(searching_result[:(len(searching_result) - 1)]):
            pos = step.set.position
            all_sets = pz.search_all_sets(pos)
            set = searching_result[j + 1].set.position
            if set == all_sets[0]:
                classes = 0
            elif set == all_sets[1]:
                classes = 1
            elif set == all_sets[2]:
                classes = 2
            elif set == all_sets[3]:
                classes = 3
            else:
                err = "ERROR to APPLY Classes in search result %s %s " % (set, all_sets)
                raise err
            print('\nTraining to - %s \n' % all_sets[classes])
            train, train_y = input_from_set(SIZE_H, SIZE_V, maps[-2], classes)
            classifier.train(
                input_fn=lambda: input_fn(train, train_y),
                steps=100)
            test, test_y = input_from_set(SIZE_H, SIZE_V, maps[-2], classes)
            eval_result = classifier.evaluate(
                input_fn=lambda: input_fn(test, test_y, training=False))
            print('\nTest set accuracy after training: {accuracy:0.3f}\n'.format(**eval_result))
            position = maps[-2]
            maps.pop()
            maps.pop()
            break

""" 
predict_x = {
    'size_h': [3, 3, 3, 3, ],
    'size_v': [3, 3, 3, 3, ],
    'position': [[1, 3, 8, 5, 7, 2, 6, 4, 0], [1, 3, 8, 5, 7, 2, 6, 0, 4], [1, 2, 3, 7, 4, 6, 5, 0, 8],  [1, 2, 3, 4, 5, 0, 7, 8, 6],],
    'toUp': [[1, 3, 8, 5, 7, 0, 6, 4, 2], [1, 3, 8, 5, 0, 2, 6, 7, 4], [1, 2, 3, 7, 0, 6, 5, 4, 8], [1, 2, 0, 4, 5, 3, 7, 8, 6],],
    'toDown': [[1, 3, 8, 5, 7, 2, 6, 4, -1], [1, 3, 8, 5, 7, 2, 6, -1, 4], [1, 2, 3, 7, 4, 6, 5, -1, 8], [1, 2, 3, 4, 5, 6, 7, 8, 0],],
    'toLeft': [[1, 3, 8, 5, 7, 2, 6, 0, 4], [1, 3, 8, 5, 7, 2, 0, 6, 4], [1, 2, 3, 7, 4, 6, 0, 5, 8],  [1, 2, 3, 4, 0, 5, 7, 8, 6],],
    'toRight': [[1, 3, 8, 5, 7, 2, 6, 4, -1], [1, 3, 8, 5, 7, 2, 6, 4, 0], [1, 2, 3, 7, 4, 6, 5, 8, 0],    [1, 2, 3, 4, 5, -1, 7, 8, 6],],
}

predictions = classifier.predict(
    input_fn=lambda: input_prediction_fn(predict_x))

for pred_dict, expec in zip(predictions, RESULT):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        RESULT[class_id], 100 * probability, expec))
"""
