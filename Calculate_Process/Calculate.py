import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

class Calculate():
    def __init__(self) -> None:
        self.Loss, self.Accuracy, self.Precision, self.Recall, self.F1, self.AUC = 0, 0, 0, 0, 0, 0
        self.Loss_Record, self.Accuracy_Record, self.Precision_Record, self.Recall_Record, self.F1_Record, self.AUC_Record = [], [], [], [], [], []
        self.History = []
        pass

    def Append_numbers(self, Loss, Accuracy, Precision, Recall, AUC, F1):
        self.Loss_Record.append(Loss)
        self.Accuracy_Record.append(Accuracy)
        self.Precision_Record.append(Precision)
        self.Recall_Record.append(Recall)
        self.F1_Record.append(F1)
        self.AUC_Record.append(AUC)
        pass

    def Calculate_Mean(self):
        Loss_Mean = np.mean(self.Loss_Record)
        Accuracy_Mean = np.mean(self.Accuracy_Record)
        Precision_Mean = np.mean(self.Precision_Record)
        Recall_Mean = np.mean(self.Recall_Record)
        F1_Mean = np.mean(self.F1_Record)
        AUC_Mean = np.mean(self.AUC_Record)

        Mean_DataFram = pd.DataFrame(
            {
                "loss" : "{:.2f}".format(Loss_Mean), 
                "precision" : "{:.2f}%".format(Precision_Mean * 100), 
                "recall" : "{:.2f}%".format(Recall_Mean * 100),
                "accuracy" : "{:.2f}%".format(Accuracy_Mean * 100), 
                "f" : "{:.2f}%".format(F1_Mean * 100), 
                "AUC" : "{:.2f}%".format(AUC_Mean * 100)
            }, index = [0]
        )
        self.History.append(Mean_DataFram)
        return Mean_DataFram
    
    def Calculate_Std(self):        
        Loss_Std = Decimal(str(np.std(self.Loss_Record))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        Accuracy_Std = Decimal(str(np.std(self.Accuracy_Record))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        Precision_Std = Decimal(str(np.std(self.Precision_Record))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        Recall_Std = Decimal(str(np.std(self.Recall_Record))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        F1_Std = Decimal(str(np.std(self.F1_Record))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        AUC_Std = Decimal(str(np.std(self.AUC_Record))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        Std_DataFram = pd.DataFrame(
            {
                "loss" : "{:.2f}".format(Loss_Std), 
                "precision" : "{:.2f}".format(Precision_Std), 
                "recall" : "{:.2f}".format(Recall_Std),
                "accuracy" : "{:.2f}".format(Accuracy_Std), 
                "f" : "{:.2f}".format(F1_Std), 
                "AUC" : "{:.2f}".format(AUC_Std)
            }, index = [0]
        )
        self.History.append(Std_DataFram)
        return Std_DataFram
    
    def Output_Style(self):
        Result = pd.DataFrame(
            {
                "loss" : "{}±{}".format(self.History[0]["loss"][0], self.History[1]["loss"][0]), 
                "precision" : "{}±{}".format(self.History[0]["precision"][0], self.History[1]["precision"][0]), 
                "recall" : "{}±{}".format(self.History[0]["recall"][0], self.History[1]["recall"][0]),
                "accuracy" : "{}±{}".format(self.History[0]["accuracy"][0], self.History[1]["accuracy"][0]), 
                "f" : "{}±{}".format(self.History[0]["f"][0], self.History[1]["f"][0]), 
                "AUC" : "{}±{}".format(self.History[0]["AUC"][0], self.History[1]["AUC"][0])
            }, index = [0]
        )
        return Result