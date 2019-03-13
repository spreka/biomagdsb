import keras.layers as kl
import keras.models as km
from os import path
from skimage import io
import numpy as np
import image_convert
import traceback
from sys import getsizeof
from descriptor_io import read_descriptor_data
from utility import find_max_idx
from utility import split_batch
from collections import OrderedDict


def get_image_path(root_dir, image_id):
    return path.join(root_dir, image_id, "images", "{}.png".format(image_id))


def get_model(fuzzy, num_of_classes, targetSize):
    model = km.Sequential()

    model.add(kl.Conv2D(32, (3, 3), input_shape=(targetSize[0], targetSize[1], 3)))
    model.add(kl.Activation('relu'))
    model.add(kl.Dropout(0.2))
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))

    model.add(kl.Conv2D(64, (3, 3)))
    model.add(kl.Activation('relu'))
    model.add(kl.Dropout(0.2))
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))

    model.add(kl.Conv2D(64, (3, 3)))
    model.add(kl.Activation('relu'))
    model.add(kl.Dropout(0.2))
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))

    model.add(kl.Flatten())
    model.add(kl.Dense(512))
    model.add(kl.Activation('relu'))
    model.add(kl.Dropout(0.4))
    model.add(kl.Dense(num_of_classes))
    model.add(kl.Activation('softmax'))

    if fuzzy:
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model


def train_model(model, train_samples, class_names, target_size):
    size = len(train_samples)
    n_epochs = 50
    one_pass_batch_size = 8
    batch_size = one_pass_batch_size*20
    print('Training model')
    for epoch in range(n_epochs):
        print('New epoch {}/{}'.format(epoch+1, n_epochs))
        batches = split_batch(train_samples, batch_size)
        for batch_id, batch in enumerate(batches):
            print('Batch id: {}/{}'.format(batch_id+1, len(batches)))
            X_train = np.zeros((batch_size, target_size[0], target_size[1], 3))
            if fuzzy:
                Y_train = np.zeros((batch_size, len(class_names)))
            else:
                Y_train = np.zeros((batch_size, 1))

            im_cont = load_data(batch.keys(), root_dir_train)

            for idx, sample in enumerate(batch.items()):
                image_id, class_values = sample
                X_train[idx] = im_cont[image_id]    # Load image here...
                if fuzzy:
                    Y_train[idx] = class_values
                else:
                    Y_train[idx] = find_max_idx(class_values)

            model.fit(X_train, Y_train, epochs=1, batch_size=one_pass_batch_size, verbose=1)


def eval_model(model, test_samples, class_names):
    hits = 0
    misses = 0
    tot_score = 0.0

    im_cont = load_data(test_samples.keys(), root_dir_train)

    for idx, sample in enumerate(test_samples.items()):
        image_id, class_values = sample

        image = im_cont[image_id]
        images = np.zeros((1, image.shape[0], image.shape[1], image.shape[2]))
        images[0] = image

        true_class_id = find_max_idx(class_values)
        true_class_name = class_names[true_class_id]

        prediction = model.predict(images)
        predicted_class_id = model.predict_classes(images)[0]
        predicted_class_name = class_names[predicted_class_id]

        tot_score += class_values[predicted_class_id]
        is_hit = (true_class_id == predicted_class_id)
        if verbose:
            print("Image: {}".format(image_id))
            print("\tTrue: {} Prediction: {} Hit: {}".format(true_class_name, predicted_class_name, is_hit))
            print("\t", prediction)
        if is_hit:
            hits += 1
        else:
            misses += 1
    n_samples = hits + misses
    hit_rate = hits / (n_samples) * 100
    exp_avg_score = tot_score/n_samples
    print("Hits: {} misses: {} hit rate: {} expected avg score: {}".format(hits, misses, hit_rate, exp_avg_score))
    return hit_rate, exp_avg_score


def load_data(sample_ids, root_dir):
    image_container = OrderedDict()
    # Load and convert all images based on the image_paths to image_list
    #print('Loading the data set.')
    data_size_bytes = 0
    total_bookkeeping = 0

    for idx, image_id in enumerate(sample_ids):
        image_raw = io.imread(get_image_path(root_dir, image_id))
        img_size_bytes = getsizeof(image_raw)
        #print('Loaded {} ({}/{}): {}x{}x{} ({}) KBytes'.format(image_id, idx + 1, len(sample_ids), *image_raw.shape, img_size_bytes/1024))
        data_size_bytes += img_size_bytes
        image_processed = image_convert.convert_image(image_raw)
        image_resized = image_convert.resize_target_size(image_processed, target_size)
        if image_processed is None:
            print("Can't read image {}!".format(image_id))

        image_container[image_id] = image_resized
        total_bookkeeping += getsizeof(image_processed) #+ getsizeof(image_resized)
    #print('Done. {} MBytes'.format(data_size_bytes/1024/1024))
    #print('Total bookkeeping: {} MBytes'.format(total_bookkeeping/1024/1024))
    return image_container


def cross_validate(samples, n_folds):
    for test_fold_id in range(n_folds):
        print('Fold: {}'.format(test_fold_id))
        m = get_model(fuzzy, len(class_names), target_size)
        train_samples = OrderedDict()
        test_samples = OrderedDict()
        for idx, sample in enumerate(samples.items()):
            image_id, class_values = sample
            if idx % n_folds == test_fold_id:
                test_samples[image_id] = class_values
            else:
                train_samples[image_id] = class_values
        train_model(m, train_samples, class_names, target_size)
        eval_model(m, test_samples, class_names)


#def save_to_disk(image_ids, images, cache_dir):
try:
    fuzzy = False
    model_file_name = "method_selector_temp4.h5"
    target_size = (256, 256)
    root_dir = "/home/etasnadi/dev/kaggle-dataset/"
    train_descriptor_file_name = "data/3/mergedTrainScores.csv"
    test_descriptor_file_name = "data/3/mergedTestScores.csv"
    root_dir_train = path.join(root_dir, "stage1_train_splitted")
    root_dir_test = path.join(root_dir, "stage1_test")
    verbose = True
    n_folds = 6

    samples, class_names = read_descriptor_data(train_descriptor_file_name)
    #image_container = load_data(samples.keys(), root_dir_train)
    cross_validate(samples, 5)
except Exception as e:
    print('Error:')
    print(e)
    traceback.print_exc()
