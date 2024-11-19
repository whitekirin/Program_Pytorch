from all_models_tools.all_model_tools import call_back
from Read_and_process_image.ReadAndProcess import Read_image_and_Process_image
from draw_tools.draw import plot_history, Confusion_Matrix_of_Two_Classification
from Load_process.Load_Indepentend import Load_Indepentend_Data
from _validation.ValidationTheEnterData import validation_the_enter_data
from Load_process.file_processing import Process_File
from merge_class.merge import merge
from draw_tools.Grad_cam import Grad_CAM
from sklearn.metrics import confusion_matrix
from Image_Process.Image_Generator import Image_generator
import pandas as pd
import time
import torch.optim as optim
from experiments.pytorch_Model import ModifiedXception
from Load_process.LoadData import Loding_Data_Root, Load_Data_Tools
import torch
from Model_Loss.Loss import Entropy_Loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchmetrics.functional import auroc
import torch.nn as nn
from torchinfo import summary
import numpy as np
from tqdm import tqdm
from torch.nn import functional

class experiments():
    def __init__(self, tools, Targets, status, Number_Of_Classes):
        '''   
            # 實驗物件

            ## 說明:
            * 用於開始訓練pytorch的物件，裡面分為數個方法，負責處理實驗過程的種種

            ## parmeter:
            * Topic_Tool: 讀取訓練、驗證、測試的資料集與Label等等的內容
            * cut_image: 呼叫切割影像物件
            * merge: 合併的物件
            * model_name: 模型名稱，告訴我我是用哪個模型(可能是預處理模型/自己設計的模型)
            * experiment_name: 實驗名稱
            * epoch: 訓練次數
            * train_batch_size: 訓練資料的batch
            * convolution_name: Grad-CAM的最後一層的名稱
            * Number_Of_Classes: Label的類別
            * Status: 選擇現在資料集的狀態
            * device: 決定使用GPU或CPU

            ## Method:
            * processing_main: 實驗物件的進入點
            * construct_model: 決定實驗用的Model
            * Training_Step: 訓練步驟，開始進行訓練驗證的部分
            * Evaluate_Model: 驗證模型的準確度
            * record_matrix_image: 劃出混淆矩陣(熱力圖)
            * record_everyTime_test_result: 記錄我單次的訓練結果並將它輸出到檔案中
        '''

        self.Topic_Tool = tools

        self.cut_image = Load_Indepentend_Data(self.Topic_Tool.Get_Data_Label(), self.Topic_Tool.Get_OneHot_Encording_Label())
        self.image_processing = Read_image_and_Process_image()
        self.merge = merge()

        self.model_name = "Xception"
        self.experiment_name = "Xception Skin to train Normal stomach cancer"
        self.generator_batch_size = 50
        self.epoch = 10000
        self.train_batch_size = 64
        self.layers = 1
        self.convolution_name = "block14_sepconv2"
        self.Number_Of_Classes = Number_Of_Classes

        self.Grad = ""
        self.Status = status
        self.Tragets = Targets
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pass

    def processing_main(self, Training_Data, counter):
        Train, Test, Validation = self.Topic_Tool.Get_Save_Roots(self.Status) # 要換不同資料集就要改
        Load_Tools = Load_Data_Tools()

        Training_Data = Load_Tools.DataLoad_Image_Root(Training_Data, self.train_batch_size)

        test = Load_Tools.Load_ImageFolder_Data(Test, "transform")
        validation = Load_Tools.Load_ImageFolder_Data(Validation, "transform")

        self.test = Load_Tools.DataLoad_Image_Root(test, 1)
        self.Validation = Load_Tools.DataLoad_Image_Root(validation, 1)

        # self.Grad = Grad_CAM(self.Topic_Tool.Get_Data_Label(), Test_labels, self.experiment_name, self.convolution_name)
        cnn_model = self.construct_model() # 呼叫讀取模型的function
        print(summary(cnn_model, input_size=(int(self.train_batch_size / 2), 3, 512, 512)))

        print("訓練開始")
        train_losses, val_losses, train_accuracies, val_accuracies = self.Training_Step(cnn_model, Training_Data, counter)
        print("訓練完成！")
        
        loss, accuracy, precision, recall, AUC, f1, True_Label, Predict_Label = self.Evaluate_Model(cnn_model)

        self.record_matrix_image(True_Label, Predict_Label, self.model_name, counter)
                
        print(self.record_everyTime_test_result(loss, accuracy, precision, recall, AUC, f1, counter, self.experiment_name)) # 紀錄當前訓練完之後的預測結果，並輸出成csv檔
            
        Losses = [train_losses, val_losses]
        Accuracyes = [train_accuracies, val_accuracies]
        plot_history(self.epoch, Losses, Accuracyes, "train" + str(counter), self.experiment_name) # 將訓練結果化成圖，並將化出來的圖丟出去儲存
        
        # self.Grad.process_main(cnn_model, counter, self.test)

        return loss, accuracy, precision, recall, AUC, f1
    
    def construct_model(self):
        '''決定我這次訓練要用哪個model'''
        cnn_model = ModifiedXception()

        if torch.cuda.device_count() > 1:
            cnn_model = nn.DataParallel(cnn_model)

        cnn_model = cnn_model.to(self.device)
        return cnn_model
    
    def Training_Step(self, model, Training, counter):
        # 定義優化器，並設定 weight_decay 參數來加入 L2 正則化
        Optimizer = optim.SGD(model.parameters(), lr=0.045, momentum = 0.9, weight_decay=0.1)
        model_path, early_stopping, scheduler = call_back(self.model_name, counter, Optimizer)

        criterion = Entropy_Loss()  # 使用自定義的損失函數
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(self.epoch):
            model.train()
            running_loss = 0.0
            all_train_preds = []
            all_train_labels = []

            epoch_iterator = tqdm(Training, desc= "Training (Epoch %d)" % epoch)


            for inputs, labels in epoch_iterator:
                labels = functional.one_hot(labels, self.Number_Of_Classes)
                # labels = np.reshape(labels, (int(labels.shape[0]), 1))
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # inputs, labels = inputs.cuda(), labels.cuda()

                Optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                Optimizer.step()
                running_loss += loss.item()

                # 收集訓練預測和標籤
                _, preds = torch.max(outputs, 1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

            Training_Loss = running_loss/len(Training)

            all_train_labels = torch.argmax(all_train_labels, 1)
            train_accuracy = accuracy_score(all_train_labels, all_train_preds)

            train_losses.append(Training_Loss)
            train_accuracies.append(train_accuracy)
            
            print(f"Epoch [{epoch+1}/{self.epoch}], Loss: {Training_Loss:.4f}, Accuracy: {train_accuracy:0.2f}", end = ' ')

            model.eval()
            val_loss = 0.0
            all_val_preds = []
            all_val_labels = []

            with torch.no_grad():
                for inputs, labels in self.Validation:
                    labels = np.reshape(labels, (int(labels.shape[0]), 1))
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # 驗證預測與標籤
                    _, preds = torch.max(outputs, 1)
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())

            # 計算驗證損失與準確率
            val_loss /= len(self.Validation)

            all_val_labels = torch.argmax(all_val_labels, 1)
            val_accuracy = accuracy_score(all_val_labels, all_val_preds)

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            print(f"Epoch [{epoch+1}/{self.epoch}], Loss: {val_loss:.4f}, Accuracy: {val_accuracy:0.2f}")

            early_stopping(val_loss, model, model_path)
            if early_stopping.early_stop:
                print("Early stopping triggered. Training stopped.")
                break

            # 學習率調整
            scheduler.step(val_loss)

        return train_losses, val_losses, train_accuracies, val_accuracies

    def Evaluate_Model(self, cnn_model):
        # 測試模型
        cnn_model.eval()
        True_Label, Predict_Label = [], []
        loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test):
                outputs = cnn_model(images)
                _, predicted = torch.max(outputs.data, 1)
                Predict_Label.extend(predicted.cpu().numpy())
                True_Label.extend(labels.cpu().numpy())

        loss /= len(self.test)

        all_val_labels = torch.argmax(all_val_labels, 1)
        accuracy = accuracy_score(True_Label, Predict_Label)
        precision = precision_score(True_Label, Predict_Label)
        recall = recall_score(True_Label, Predict_Label)
        AUC = auroc(True_Label, Predict_Label)
        f1 = f1_score(True_Label, Predict_Label)
        return loss, accuracy, precision, recall, AUC, f1, True_Label, Predict_Label


    def record_matrix_image(self, True_Labels, Predict_Labels, model_name, index):
        '''劃出混淆矩陣(熱力圖)'''
        # 計算混淆矩陣
        matrix = confusion_matrix(True_Labels, Predict_Labels)
        Confusion_Matrix_of_Two_Classification(model_name, matrix, index) # 呼叫畫出confusion matrix的function

        return matrix.real

    def record_everyTime_test_result(self, loss, accuracy, precision, recall, auc, f, indexs, model_name):
        '''記錄我單次的訓練結果並將它輸出到檔案中'''
        File = Process_File()

        Dataframe = pd.DataFrame(
                {
                    "model_name" : str(model_name),
                    "loss" : "{:.2f}".format(loss), 
                    "precision" : "{:.2f}%".format(precision * 100), 
                    "recall" : "{:.2f}%".format(recall * 100),
                    "accuracy" : "{:.2f}%".format(accuracy * 100), 
                    "f" : "{:.2f}%".format(f * 100), 
                    "AUC" : "{:.2f}%".format(auc * 100)
                }, index = [indexs])
        File.Save_CSV_File("train_result", Dataframe)
        # File.Save_TXT_File("Matrix_Result : " + str(Matrix), model_name + "_train" + str(indexs))

        return Dataframe

