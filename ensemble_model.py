import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
import keras
import tensorflow as tf
from keras.models import load_model
from keras.layers import Flatten, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications import DenseNet121
from keras.applications import EfficientNetB2

def load(weights_path,w1,w2,w3,w4,w5):
    # Load pre-trained models
    all_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    with tf.keras.utils.custom_object_scope({'Functional': tf.keras.models.Model}):
        # InceptionResNetV2 model
        img_in = Input(shape=(224, 224, 3))  
        inception_resnet_v2 = InceptionResNetV2(include_top= False , 
                                                weights='imagenet',      
                                                input_tensor= img_in, 
                                                input_shape= (224, 224, 3),
                                                pooling ='avg')
        x = inception_resnet_v2.output
        predictions = Dense(len(all_labels), activation="sigmoid", name="predictions_1")(x)  
        inception_resnet_v2 = Model(inputs=img_in, outputs=predictions)
        inception_resnet_v2.load_weights(w1)
        inception_resnet_v2._name = 'inception_resnet_v2'

        # MobileNet model
        mobilenet = MobileNet(include_top= False , 
                                weights='imagenet',      
                                input_tensor= img_in, 
                                input_shape= (224, 224, 3),
                                pooling ='avg') 
        x = mobilenet.output
        predictions = Dense(len(all_labels), activation="sigmoid", name="predictions_2")(x)
        mobilenet = Model(inputs=img_in, outputs=predictions)
        mobilenet.load_weights(w2)
        mobilenet._name = 'mobilenet'

        # DenseNet121 model
        densenet121 = DenseNet121(include_top=False,
                                weights='imagenet',
                                input_tensor=img_in,
                                input_shape=(224, 224, 3),
                                pooling='avg')
        x = densenet121.output
        predictions = Dense(len(all_labels), activation="sigmoid", name="predictions_3")(x)
        densenet121 = Model(inputs=img_in, outputs=predictions)
        densenet121.load_weights(w3)
        densenet121._name = 'densenet121'

        # EfficientNetB2 model
        efficientnetb2 = EfficientNetB2(include_top=False,
                                        weights='imagenet',
                                        input_tensor=img_in,
                                        input_shape=(224, 224, 3),
                                        pooling='avg')
        x = efficientnetb2.output
        predictions = Dense(len(all_labels), activation="sigmoid", name="predictions_4")(x)
        efficientnetb2 = Model(inputs=img_in, outputs=predictions)
        efficientnetb2.load_weights(w4)
        efficientnetb2._name = 'efficientnetb2'

        # InceptionV3 model
        inceptionv3 = InceptionV3(include_top=False,
                                weights=None,
                                input_tensor=img_in,
                                input_shape=(224, 224, 3),
                                pooling='avg')
        x = inceptionv3.output
        predictions = Dense(len(all_labels), activation="sigmoid", name="predictions_5")(x)
        inceptionv3 = Model(inputs=img_in, outputs=predictions)
        inceptionv3.load_weights(w5)
        inceptionv3._name = 'inceptionv3'

    # Ensemble model
    x1 = inception_resnet_v2(img_in)
    x1 = Flatten()(x1)
    x1 = Dense(256, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)

    x2 = mobilenet(img_in)
    x2 = Flatten()(x2)
    x2 = Dense(256, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)

    x3 = densenet121(img_in)
    x3 = Flatten()(x3)
    x3 = Dense(256, activation='relu')(x3)
    x3 = Dropout(0.5)(x3)

    x4 = efficientnetb2(img_in)
    x4 = Flatten()(x4)
    x4 = Dense(256, activation='relu')(x4)
    x4 = Dropout(0.5)(x4)

    x5 = inceptionv3(img_in)
    x5 = Flatten()(x5)
    x5 = Dense(256, activation='relu')(x5)
    x5 = Dropout(0.5)(x5)

    combined = tf.keras.layers.concatenate([x1, x2, x3, x4, x5])

    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.5)(x)

    predictions = Dense(len(all_labels), activation="sigmoid", name="final_predictions")(x)

    ensemble_model = Model(inputs=img_in, outputs=predictions)

    optimizer = Adam(learning_rate=0.001)
    ensemble_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[keras.metrics.binary_accuracy])

    ensemble_model.load_weights(weights_path)
    return ensemble_model