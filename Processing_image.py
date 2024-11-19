from merge_class.merge import merge
from Load_process.Loading_Tools import Load_Data_Prepare
from Load_process.LoadData import Loding_Data_Root
from Training_Tools.Tools import Tool
from Read_and_process_image.ReadAndProcess import Read_image_and_Process_image
from matplotlib import pyplot as plt

if __name__ == "__main__":
    Merge = merge()
    read = Read_image_and_Process_image()
    tool = Tool()
    Prepare = Load_Data_Prepare()
    
    tool.Set_Labels()
    tool.Set_Save_Roots()
    Labels = tool.Get_Data_Label()
    Trainig_Root, Testing_Root, Validation_Root = tool.Get_Save_Roots(2)

    load = Loding_Data_Root(Labels, Trainig_Root, "")
    Data_Root = load.get_Image_data_roots(Trainig_Root)

    # 將資料做成Dict的資料型態
    Prepare.Set_Final_Dict_Data(Labels, Data_Root, [[], []], 2)
    Final_Dict_Data = Prepare.Get_Final_Data_Dict()
    keys = list(Final_Dict_Data.keys())

    training_data = Merge.merge_all_image_data(Final_Dict_Data[keys[0]], Final_Dict_Data[keys[1]]) # 將訓練資料合併成一個list

    Image = read.Data_Augmentation_Image(training_data)
    plt.imshow(Image[0])
    plt.show()

