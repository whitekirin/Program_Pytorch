from convolution_model_tools.convolution_2D_tools import model_2D_tool
from dense_model_tools.dense_tools import model_Dense_Layer
from all_models_tools.all_model_tools import add_optimizers_function, add_Activative, add_dropout, call_back
from keras.activations import softmax, sigmoid
from keras.applications import VGG19, ResNet50, NASNetLarge, DenseNet201, Xception
from keras.applications.efficientnet_v2 import EfficientNetV2L
from keras.layers import BatchNormalization, Flatten, GlobalAveragePooling2D, MaxPooling2D, Dense, Conv2D, Dropout, TimeDistributed, LSTM, Input
from keras import regularizers

def one_layer_cnn_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    img_Input = tools.add_2D_input()
    x = tools.add_Convolution2D(img_Input, 32)
    x = add_Activative(x)
    x = tools.add_MaxPooling(x)

    x = tools.add_Convolution2D(x, 64)
    x = add_Activative(x)
    x = tools.add_MaxPooling(x)

    flatter = tools.add_flatten(x)

    dense = dense_tool.add_dense(64, flatter)
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(32, dense)
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, softmax)
    return img_Input, dense

def find_example_cnn_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    img_Input = tools.add_2D_input()

    x = tools.add_Convolution2D(img_Input, 16)
    x = add_Activative(x)
    x = add_dropout(x, 0.25)

    x = tools.add_Convolution2D(x, 32)
    x = add_Activative(x)
    x = add_dropout(x, 0.25)

    x = tools.add_MaxPooling(x)

    x = tools.add_Convolution2D(x, 64)
    x = add_Activative(x)
    x = add_dropout(x, 0.25)

    x = tools.add_MaxPooling(x)

    x = tools.add_Convolution2D(x, 128)
    x = add_Activative(x)
    x = add_dropout(x, 0.25)

    x = tools.add_MaxPooling(x)

    flatter = tools.add_flatten(x)

    dense = dense_tool.add_dense(64, flatter)
    dense = add_Activative(dense)
    dense = add_dropout(dense, 0.25)
    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, sigmoid)

    return img_Input, dense

def change_example_cnn_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    img_Input = tools.add_2D_input()

    x = tools.add_Convolution2D(img_Input, 16)
    x = add_Activative(x)
    x = tools.add_batchnomlization(x)

    x = tools.add_Convolution2D(x, 32)
    x = add_Activative(x)
    x = tools.add_batchnomlization(x)

    x = tools.add_MaxPooling(x)

    x = tools.add_Convolution2D(x, 64)
    x = add_Activative(x)
    x = tools.add_batchnomlization(x)

    x = tools.add_MaxPooling(x)

    x = tools.add_Convolution2D(x, 128)
    x = add_Activative(x)
    x = tools.add_batchnomlization(x)

    x = tools.add_MaxPooling(x)

    flatter = tools.add_flatten(x)

    dense = dense_tool.add_dense(64, flatter)
    dense = add_Activative(dense)
    dense = add_dropout(dense, 0.3)
    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, softmax)

    return img_Input, dense

def two_convolution_cnn_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    img_Input = tools.add_2D_input()
    x = tools.add_two_floors_convolution2D(img_Input, 32)
    x = tools.add_MaxPooling(x)

    x = tools.add_two_floors_convolution2D(x, 64)
    x = tools.add_MaxPooling(x)

    flatter = tools.add_flatten(x)

    dense = dense_tool.add_dense(64, flatter)
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(32, dense)
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, softmax)
    return img_Input, dense  

def VGG19_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    vgg19 = VGG19(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = tools.add_flatten(vgg19.output)
    dense = dense_tool.add_dense(64, flatten)
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, softmax)

    return vgg19, dense  

def Resnet50_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    vgg19 = ResNet50(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = tools.add_flatten(vgg19.output)
    dense = dense_tool.add_dense(64, flatten)
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, softmax)

    return vgg19, dense  

def DenseNet201_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    Densenet201 = DenseNet201(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = tools.add_flatten(Densenet201.output)
    dense = dense_tool.add_dense(64, flatten)
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, softmax)

    return Densenet201, dense

def Xception_model():
    xception = Xception(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = Flatten()(xception.output)
    dense = Dense(units = 64, activation = "relu")(flatten)
    dense = Dense(units = 7, activation = "softmax")(dense)

    return xception, dense

def cnn_LSTM():
    head = Input(shape = (150, 150, 3))
    inputs = Conv2D(filters = 64, strides = 1, kernel_size = (3, 3), padding = "same", activation = "relu")(head)
    inputs = Conv2D(filters = 64, strides = 1, kernel_size = (3, 3), padding = "same", activation = "relu")(inputs)
    inputs = MaxPooling2D(strides = 2, pool_size = (2, 2))(inputs)
    inputs = Dropout(0.25)(inputs)

    inputs = Conv2D(filters = 128, strides = 1, kernel_size = (3, 3), padding = "same", activation = "relu")(inputs)
    inputs = Conv2D(filters = 128, strides = 1, kernel_size = (3, 3), padding = "same", activation = "relu")(inputs)
    inputs = MaxPooling2D(strides = 2, pool_size = (2, 2))(inputs)
    inputs = Dropout(0.25)(inputs)

    inputs = Conv2D(filters = 256, strides = 1, kernel_size = (3, 3), padding = "same", activation = "relu")(inputs)
    inputs = Conv2D(filters = 256, strides = 1, kernel_size = (3, 3), padding = "same", activation = "relu")(inputs)
    inputs = MaxPooling2D(strides = 2, pool_size = (2, 2))(inputs)
    inputs = Dropout(0.25)(inputs)

    inputs = Conv2D(filters = 512, strides = 1, kernel_size = (3, 3), padding = "same", activation = "relu")(inputs)
    inputs = Conv2D(filters = 512, strides = 1, kernel_size = (3, 3), padding = "same", activation = "relu")(inputs)
    inputs = Conv2D(filters = 512, strides = 1, kernel_size = (3, 3), padding = "same", activation = "relu")(inputs)
    inputs = MaxPooling2D(strides = 2, pool_size = (2, 2))(inputs)
    inputs = Dropout(0.25)(inputs)

    inputs = Conv2D(filters = 512, strides = 1, kernel_size = (3, 3), padding = "same", activation = "relu")(inputs)
    inputs = Conv2D(filters = 512, strides = 1, kernel_size = (3, 3), padding = "same", activation = "relu")(inputs)
    inputs = Conv2D(filters = 512, strides = 1, kernel_size = (3, 3), padding = "same", activation = "relu")(inputs)
    inputs = MaxPooling2D(strides = 2, pool_size = (2, 2))(inputs)
    inputs = Dropout(0.25)(inputs)
    inputs = TimeDistributed(Flatten())(inputs)

    inputs = LSTM(units = 49)(inputs)
    inputs = Dense(units = 64)(inputs)
    output = Dense(units = 7, activation = "softmax")(inputs)

    return head, output

def add_regularizers_L1(): # 比較正規化
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    Resnet50 = ResNet50(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = tools.add_flatten(Resnet50.output)
    dense = dense_tool.add_regularizer_dense(64, flatten, regularizers.L1())
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, softmax)

    return Resnet50, dense

def add_regularizers_L2(): # 比較正規化
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    Resnet50 = ResNet50(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = tools.add_flatten(Resnet50.output)
    dense = dense_tool.add_regularizer_dense(64, flatten, regularizers.L2())
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, softmax)

    return Resnet50, dense

def add_regularizers_L1L2(): # 比較正規化
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    Resnet50 = ResNet50(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = tools.add_flatten(Resnet50.output)
    dense = dense_tool.add_regularizer_dense(64, flatten, regularizers.L1L2())
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, softmax)

    return Resnet50, dense

def add_layers1_L2(Dense_layers): # 比較正規化
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()
    layers = 32

    Densenet201 = DenseNet201(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = tools.add_flatten(Densenet201.output)

    for layer in range(Dense_layers):
        dense = dense_tool.add_regularizer_kernel_dense(unit = layers, input_data = flatten, regularizer = regularizers.L2())
        dense = add_Activative(dense)
        layers *= 2

    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, softmax)

    return Densenet201, dense

def add_layers_another_L2(Dense_layers, layers): # 比較正規化
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    Densenet201 = DenseNet201(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = tools.add_flatten(Densenet201.output)

    for layer in range(Dense_layers):
        dense = dense_tool.add_regularizer_dense(unit = layers, input_data = flatten, regularizer = regularizers.L2())
        dense = add_Activative(dense)
        layers /= 2

    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, softmax)

    return Densenet201, dense

def add_bias_regularizers(): # 比較正規化
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    Resnet50 = ResNet50(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = tools.add_flatten(Resnet50.output)
    dense = dense_tool.add_regularizer_bias_dense(64, flatten, regularizers.L2())
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(7, dense)
    dense = add_Activative(dense, softmax)

    return Resnet50, dense