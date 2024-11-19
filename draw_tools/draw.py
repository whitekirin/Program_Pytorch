from matplotlib import pyplot as plt
import seaborn as sns
import datetime
import matplotlib.figure as figure
import matplotlib.backends.backend_agg as agg
from Load_process.file_processing import Process_File

def plot_history(Epochs, Losses, Accuracys, file_name, model_name):
    File = Process_File()

    plt.figure(figsize=(16,4))
    plt.subplot(1,2,1)
    plt.plot(range(1, Epochs + 1), Losses[0])
    plt.plot(range(1, Epochs + 1), Losses[1])
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train','Validation'], loc='upper left')
    plt.title('Model Accuracy')

    plt.subplot(1,2,2)
    plt.plot(range(1, Epochs + 1), Accuracys[0])
    plt.plot(range(1, Epochs + 1), Accuracys[1])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train','Validation'], loc='upper left')
    plt.title('Model Loss')

    model_dir = '../Result/save_the_train_image( ' + str(datetime.date.today()) + " )"
    File.JudgeRoot_MakeDir(model_dir)
    modelfiles = File.Make_Save_Root(str(model_name) + " " + str(file_name) + ".png", model_dir)
    plt.savefig(modelfiles)
    plt.close("all")      # 關閉圖表

def draw_heatmap(matrix, model_name, index): # 二分類以上混淆矩陣做法
    File = Process_File()

    # 创建热图
    fig = figure.Figure(figsize=(6, 4))
    canvas = agg.FigureCanvasAgg(fig)
    Ax = fig.add_subplot(111)
    sns.heatmap(matrix, square = True, annot = True, fmt = 'd', linecolor = 'white', cmap = "Purples", ax = Ax)#画热力图，cmap表示设定的颜色集

    model_dir = '../Result/model_matrix_image ( ' + str(datetime.date.today()) + " )"
    File.JudgeRoot_MakeDir(model_dir)
    modelfiles = File.Make_Save_Root(str(model_name) + "-" + str(index) + ".png", model_dir)

    # confusion.figure.savefig(modelfiles)
    # 设置图像参数
    Ax.set_title(str(model_name) + " confusion matrix")
    Ax.set_xlabel("X-Predict label of the model")
    Ax.set_ylabel("Y-True label of the model")

    # 保存图像到文件中
    canvas.print_figure(modelfiles)

def Confusion_Matrix_of_Two_Classification(Model_Name, Matrix, index):
    File = Process_File()

    fx = sns.heatmap(Matrix, annot=True, cmap='turbo')

    # labels the title and x, y axis of plot
    fx.set_title('Plotting Confusion Matrix using Seaborn\n\n')
    fx.set_xlabel('answer Values ')
    fx.set_ylabel('Predicted Values')

    # labels the boxes
    fx.xaxis.set_ticklabels(['False','True'])
    fx.yaxis.set_ticklabels(['False','True'])

    model_dir = '../Result/model_matrix_image ( ' + str(datetime.date.today()) + " )"
    File.JudgeRoot_MakeDir(model_dir)
    modelfiles = File.Make_Save_Root(str(Model_Name) + "-" + str(index) + ".png", model_dir)

    plt.savefig(modelfiles)
    plt.close("all")      # 關閉圖表

    pass