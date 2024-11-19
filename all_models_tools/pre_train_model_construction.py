from all_models_tools.all_model_tools import attention_block
from keras.activations import softmax, sigmoid
from keras.applications import VGG16,VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, NASNetLarge, Xception 
from keras.layers import GlobalAveragePooling2D, Dense, Flatten
from keras import regularizers
from keras.layers import Add
from application.Xception_indepentment import Xception_indepentment   

def Original_VGG19_Model():
    vgg19 = VGG19(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(vgg19.output)
    dense = Dense(units = 4096, activation = "relu")(GAP)
    dense = Dense(units = 4096, activation = "relu")(dense)
    output = Dense(units = 2, activation = "softmax")(dense)

    return vgg19.input, output

def Original_ResNet50_model():
    xception = ResNet50(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(xception.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return xception.input, dense

def Original_NASNetLarge_model():
    nasnetlarge = NASNetLarge(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(nasnetlarge.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return nasnetlarge.input, dense

def Original_DenseNet121_model():
    Densenet201 = DenseNet121(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(Densenet201.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return Densenet201.input, dense

def Original_Xception_model():
    xception = Xception(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(xception.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return xception.input, dense

def Original_VGG16_Model():
    vgg16 = VGG16(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    flatten = Flatten()(vgg16.output)
    dense = Dense(units = 4096, activation = "relu")(flatten)
    dense = Dense(units = 4096, activation = "relu")(dense)
    output = Dense(units = 2, activation = "softmax")(dense)

    return vgg16.input, output

def Original_ResNet50v2_model():
    resnet50v2 = ResNet50V2(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(resnet50v2.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return resnet50v2.input, dense

def Original_ResNet101_model():
    resnet101 = ResNet101(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(resnet101.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return resnet101.input, dense

def Original_ResNet101V2_model():
    resnet101v2 = ResNet101V2(include_top = False, weights = "imagenet", input_shape = (512, 512, 3))
    GAP = GlobalAveragePooling2D()(resnet101v2.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return resnet101v2.input, dense

def Original_ResNet152_model():
    resnet152 = ResNet152(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(resnet152.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return resnet152.input, dense

def Original_ResNet152V2_model():
    resnet152v2 = ResNet152V2(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(resnet152v2.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return resnet152v2.input, dense

def Original_InceptionV3_model():
    inceptionv3 = InceptionV3(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(inceptionv3.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return inceptionv3.input, dense

def Original_InceptionResNetV2_model():
    inceptionResnetv2 = InceptionResNetV2(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(inceptionResnetv2.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return inceptionResnetv2.input, dense

def Original_MobileNet_model():
    mobilenet = MobileNet(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(mobilenet.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return mobilenet.input, dense

def Original_MobileNetV2_model():
    mobilenetv2 = MobileNetV2(include_top = False, weights = "imagenet", input_shape = (200, 200, 3))
    GAP = GlobalAveragePooling2D()(mobilenetv2.output)
    dense = Dense(units = 2, activation = "softmax")(GAP)

    return mobilenetv2.input, dense