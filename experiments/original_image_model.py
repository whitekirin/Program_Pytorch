from convolution_model_tools.convolution_2D_tools import model_2D_tool
from dense_model_tools.dense_tools import model_Dense_Layer
from all_models_tools.all_model_tools import add_Activative, add_dropout
from keras.activations import softmax, sigmoid
from keras.applications import VGG19, ResNet50, InceptionResNetV2, Xception, DenseNet169, EfficientNetV2L

def original_VGG19_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    vgg19 = VGG19(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    GAP = tools.add_globalAveragePooling(vgg19.output)
    # flatten = tools.add_flatten(vgg19.output)
    dense = dense_tool.add_dense(256, GAP)
    # dense = add_Activative(dense)
    dense = dense_tool.add_dense(4, dense)
    dense = add_Activative(dense, softmax)

    return vgg19.input, dense

def original_Resnet50_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    resnet50 = ResNet50(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    GAP = tools.add_globalAveragePooling(resnet50.output)
    dense = dense_tool.add_dense(256, GAP)
    dense = dense_tool.add_dense(4, dense)
    dense = add_Activative(dense, softmax)

    return resnet50, dense

def original_InceptionResNetV2_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    inceptionresnetv2 = InceptionResNetV2(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = tools.add_flatten(inceptionresnetv2.output)
    dense = dense_tool.add_dense(256, flatten)
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(4, dense)
    dense = add_Activative(dense, softmax)

    return inceptionresnetv2.input, dense

def original_Xception_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    xception = Xception(include_top = False, weights = "imagenet", input_shape = (150, 150, 3))
    GAP = tools.add_globalAveragePooling(xception.output)
    dense = dense_tool.add_dense(256, GAP)
    dense = dense_tool.add_dense(4, dense)
    dense = add_Activative(dense, softmax)

    return xception, dense

def original_EfficientNetV2L_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    EfficientNet_V2L = EfficientNetV2L(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = tools.add_flatten(EfficientNet_V2L.output)
    dense = dense_tool.add_dense(256, flatten)
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(4, dense)
    dense = add_Activative(dense, softmax)

    return EfficientNet_V2L.input, dense

def original_DenseNet169_model():
    tools = model_2D_tool()
    dense_tool = model_Dense_Layer()

    Densenet169 = DenseNet169(include_top = False, weights = "imagenet", input_shape = (120, 120, 3))
    flatten = tools.add_flatten(Densenet169.output)
    dense = dense_tool.add_dense(256, flatten)
    dense = add_Activative(dense)
    dense = dense_tool.add_dense(4, dense)
    dense = add_Activative(dense, softmax)

    return Densenet169.input, dense