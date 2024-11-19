from Read_and_process_image.ReadAndProcess import Read_image_and_Process_image
from sklearn.model_selection import train_test_split
from model_data_processing.processing import shuffle_data
from merge_class.merge import merge
from Read_and_process_image.ReadAndProcess import Read_image_and_Process_image
from Load_process.LoadData import Load_Data_Prepare, Process_File, Load_Data_Tools
import shutil

class Cut_Indepentend_Data():
    def __init__(self, Training_Root, Labels) -> None:
        self.Training_Root = Training_Root
        self.Labels = Labels
        pass
    def IndependentData_main(self, Indepentend_Data_Root, Test_Size): # 製作獨立資料
        Prepare = Load_Data_Prepare()
        Load_Tool = Load_Data_Tools()
        Prepare.Set_Data_Content([], len(self.Labels))
        Prepare.Set_Data_Dictionary(self.Labels, Prepare.Get_Data_Content(), 2)
        Get_Data_Dict_Content = Prepare.Get_Data_Dict()
        get_all_image_data = Load_Tool.get_data_root(self.Training_Root, Get_Data_Dict_Content, self.Labels)

        self.Cut_Of_Independent_Data(get_all_image_data, Indepentend_Data_Root, Test_Size)

    def Balance_Cut_Of_Independent_Data(self, Independent_Dict_Data_Content, Test_Size):
        image_processing = Read_image_and_Process_image()
        Prepare = Load_Data_Prepare()
        Prepare.Set_Data_Content([], len(self.Labels))
        Prepare.Set_Data_Dictionary(self.Labels, Prepare.Get_Data_Content(), 2)
        Indepentend_Content = Prepare.Get_Data_Dictionary()

        for cut_TotalTestData_roots_label in self.Labels: # 將資料的label一個一個讀出來                    
            label = image_processing.make_label_list(len(Independent_Dict_Data_Content[cut_TotalTestData_roots_label]), 0) # 製作假的label
            tmp = list(Cut_Data(Independent_Dict_Data_Content[cut_TotalTestData_roots_label], label, Test_Size)) # 切割出特定數量的資料
            Indepentend_Content[cut_TotalTestData_roots_label] = [tmp[0], tmp[1]]

        return Indepentend_Content

    def Cut_Of_Independent_Data(self, Independent_Dict_Data_Content, IndependentDataRoot, Test_Size):
        '''切割獨立資料(e.g. Validation、training)'''
        image_processing = Read_image_and_Process_image()
        Prepaer = Load_Data_Prepare()
        File = Process_File()
        i = 0

        if not File.Judge_File_Exist(IndependentDataRoot): #若要儲存的資料夾不存在
            for cut_TotalTestData_roots_label in self.Labels: # 將資料的label一個一個讀出來
                root = File.Make_Save_Root(cut_TotalTestData_roots_label, IndependentDataRoot) # 合併成一個路徑
                File.Make_Dir(root) # 建檔
                    
                label = image_processing.make_label_list(len(Independent_Dict_Data_Content[cut_TotalTestData_roots_label]), 0) # 製作假的label
                cut_data = Cut_Data(Independent_Dict_Data_Content[cut_TotalTestData_roots_label], label, Test_Size) # 切割出特定數量的資料
                
                for data in cut_data[1]:
                    shutil.move(data, root) # 移動資料進新的資料夾
                i += 1
        
def Cut_Data(data, label, TestSize = 0.2, random = 2022):
        '''切割資料集'''
        train, test, train_label, test_label = train_test_split(data, label, test_size = TestSize, random_state = random)
        return (train, test, train_label, test_label)