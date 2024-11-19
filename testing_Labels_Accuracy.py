from experiments.experiment import experiments
from concurrent.futures import ProcessPoolExecutor
from loadData_and_MakeImageGenerator.load_and_ImageGenerator import Load_ImageGenerator
from Read_and_process_image.ReadAndProcess import Read_image_and_Process_image
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from draw_tools.draw import draw_heatmap
from Load_process.LoadData import Loding_Data_Root
import os
import seaborn as sns
import datetime
from Load_process.file_processing import judge_file_exist, make_dir, make_save_root
from matplotlib import pyplot as plt
from Load_process.Load_Indepentend import Load_Indepentend_Data

def draw(matrix, model_name, index):
    # Using Seaborn heatmap to create the plot

    fx = sns.heatmap(matrix, annot=True, cmap='turbo')
    # labels the title and x, y axis of plot
    fx.set_title('Plotting Confusion Matrix using Seaborn\n\n')
    fx.set_xlabel('Predicted Values')
    fx.set_ylabel('answer Values ')
    # labels the boxes
    fx.xaxis.set_ticklabels(['False','True'])
    fx.yaxis.set_ticklabels(['False','True'])

    model_dir = '../../Model_Confusion_matrix/model_matrix_image ( ' + str(datetime.date.today()) + " )/" + model_name
    if not judge_file_exist(model_dir):
        make_dir(model_dir)
    modelfiles = make_save_root(str(model_name) + "-" + str(index) + ".png", model_dir)
    plt.savefig(modelfiles)
    plt.close("all")      # 關閉圖表

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor: ## 默认为1
        print('TensorFlow version:', tf.__version__)
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        print(physical_devices)
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        os.environ["CUDA_VISIBLE_DEVICES"]='0'

        load = Loding_Data_Root()
        experiment = experiments()
        image_processing = Read_image_and_Process_image()
        cut_image = Load_Indepentend_Data()

        model = experiment.construct_model()
        model_dir = '../../best_model( 2023-10-17 )-2.h5' # 這是一個儲存模型權重的路徑，每一個模型都有一個自己權重儲存的檔
        if os.path.exists(model_dir): # 如果這個檔案存在
            model.load_weights(model_dir) # 將模型權重讀出來
            print("讀出權重\n")
        
        for times in range(5):
            name = ["BP", "PF", "PV", "Chickenpox", "Monkeypox", "Normal", "Another"]        
            cut_image.process_main() # 呼叫處理test Data與Validation Data的function
            test, test_label = cut_image.test, cut_image.test_label

            total_data = [[], [], [], [], [], [], []]
            total_labels = [[], [], [], [], [], [], []]
            start = 0
            end = 22
            for k in range(7):
                for i in range(start, end):
                    total_data[k].append(test[i])
                    total_labels[k].append(test_label[i])
                    
                total_data[k], total_labels[k] = image_processing.image_data_processing(total_data[k], total_labels[k])
                total_data[k] = image_processing.normalization(total_data[k])

                start = end
                end += 22
                

            j = 0
            for total_label in range(7):
                result = model.predict(total_data[j]) # 利用predict function來預測結果
                result = np.argmax(result, axis = 1)  # 將預測出來的結果從one-hot encoding轉成label-encoding
                y_test = np.argmax(total_labels[j], axis = 1)
                    
                print(name[j] + str(result), "\n")

                y_pre = []
                for i in range(len(result)):
                    if result[i] != j:
                        result[i] = 0
                    else:
                        result[i] = 1

                    y_test[i] = 1

                matrix = confusion_matrix(y_test, result, labels = [0, 1]) # 丟入confusion matrix的function中，以形成混淆矩陣
                draw(matrix, name[j], times) # 呼叫畫出confusion matrix的function
                tn, fp, fn, tp = matrix.ravel()

                accuracy = (tn + tp) / (tn + fp + fn + tp)
                print(name[j] + " 權重為: ", accuracy)

                experiment.record_everyTime_test_result(0, accuracy, 0, 0, 0, 0, times, name[j])

                j += 1
