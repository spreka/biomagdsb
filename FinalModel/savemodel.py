import tensorflow

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.Session(config=config)


import os
import os.path

import sys
sys.path.append('/home/kriston/kaggle/Mask_RCNN/mrcnn')

import model
import config
import keras
import keras.backend
import keras.layers
import keras.engine
import keras.models


print("Usage", sys.argv[0], "settings.json")


class TempConfig(config.Config):
    NAME = "nuclei"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # background + nucleus
    TRAIN_ROIS_PER_IMAGE = 512
    STEPS_PER_EPOCH = 5000 # check mask_train for the final value
    VALIDATION_STEPS = 50
    DETECTION_MAX_INSTANCES = 512
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.35
    RPN_NMS_THRESHOLD = 0.55
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_DIM = 1024


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tensorflow.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tensorflow.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def SaveModel(pInModelPath, pOutModelDir):
        config = TempConfig()
        # show config
        config.display()

        inferencemdl = model.MaskRCNN(mode="inference", config=config, model_dir=os.path.dirname(pInModelPath))
        inferencemdl.load_weights(pInModelPath, by_name=True)
        inference_frozen_graph = freeze_session(keras.backend.get_session(), output_names=[out.op.name for out in inferencemdl.keras_model.outputs])
        tensorflow.train.write_graph(inference_frozen_graph, pOutModelDir, "maskrcnn_inference_model.pb", as_text=False)
        tensorflow.train.write_graph(inference_frozen_graph, pOutModelDir, "maskrcnn_inference_model.pbtxt", as_text=True)

        print("Inference saved")

        keras.backend.clear_session()

        trainmdl = model.MaskRCNN(mode="training", config=config, model_dir=os.path.dirname(pInModelPath))
        trainmdl.load_weights(pInModelPath, by_name=True)
        train_frozen_graph = freeze_session(keras.backend.get_session(), output_names=[out.op.name for out in trainmdl.keras_model.outputs])
        tensorflow.train.write_graph(train_frozen_graph, pOutModelDir, "maskrcnn_train_model.pb", as_text=False)
        tensorflow.train.write_graph(train_frozen_graph, pOutModelDir, "maskrcnn_train_model.pbtxt", as_text=True)

        print("Train saved")

SaveModel(sys.argv[1], sys.argv[2])



