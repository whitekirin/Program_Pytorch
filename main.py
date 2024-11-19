from experiments.experiment import experiments
from Image_Process.load_and_ImageGenerator import Load_ImageGenerator
from Read_and_process_image.ReadAndProcess import Read_image_and_Process_image
from Training_Tools.Tools import Tool
from Load_process.LoadData import Load_Data_Prepare, Load_Data_Tools
from Calculate_Process.Calculate import Calculate
from merge_class.merge import merge
import torch

if __name__ == "__main__":
    # 測試GPU是否可用
    flag = torch.cuda.is_available()
    if not flag:
        print("CUDA不可用\n")
    else:
        print(f"CUDA可用，數量為{torch.cuda.device_count()}\n")

    Load_Tools = Load_Data_Tools()
    Status = 2 # 決定要使用什麼資料集
    # 要換不同資料集就要改
    tool = Tool()
    tool.Set_Labels()
    tool.Set_Save_Roots()

    Labels = tool.Get_Data_Label()
    Trainig_Root, Testing_Root, Validation_Root = tool.Get_Save_Roots(Status) # 一般的
    Generator_Root = tool.Get_Generator_Save_Roots(Status)

    # 取得One-hot encording 的資料
    tool.Set_OneHotEncording(Labels)
    Encording_Label = tool.Get_OneHot_Encording_Label()
    Label_Length = len(Labels)

    Gneerator_size = 0
    Prepare = Load_Data_Prepare()
    loading_data = Load_ImageGenerator(Trainig_Root, Testing_Root, Validation_Root, Generator_Root, Labels)
    image_processing = Read_image_and_Process_image()
    Merge = merge()
    Calculate_Tool = Calculate()
 
    counter = 5
    Number_Of_Count = 2
    
    for i in range(0, counter, 1): # 做規定次數的訓練
        # 讀取資料
        Image_Data, Total_Data_List = loading_data.process_main(Label_Length)
        Training_Data = Load_Tools.Get_Balance_Data(Image_Data, Total_Data_List)
        experiment = experiments(tool, Total_Data_List, Status, Number_Of_Count)
                 
        loss, accuracy, precision, recall, AUC, f = experiment.processing_main(Training_Data, i) # 執行訓練方法
        Calculate_Tool.Append_numbers(loss, accuracy, precision, recall, AUC, f)

    print("實驗結果")
    print("--------------------------------------------")
    print("平均值: ")
    print(Calculate_Tool.Calculate_Mean())
    print("標準差: ")
    print(Calculate_Tool.Calculate_Std())
    print("結果: ")
    print(Calculate_Tool.Output_Style())
    