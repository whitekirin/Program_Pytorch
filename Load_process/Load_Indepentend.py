from Read_and_process_image.ReadAndProcess import Read_image_and_Process_image
from model_data_processing.processing import shuffle_data
from merge_class.merge import merge
from Read_and_process_image.ReadAndProcess import Read_image_and_Process_image
from Load_process.LoadData import Load_Data_Prepare, Load_Data_Tools

class Load_Indepentend_Data():
    def __init__(self, Labels, OneHot_Encording):
        '''
        影像切割物件
        label有2類,會將其轉成one-hot-encoding的形式
            [0, 1] = NPC_negative
            [1, 0] = NPC_positive
        '''
        self.merge = merge()
        self.Labels = Labels
        self.OneHot_Encording = OneHot_Encording
        pass

    def process_main(self, Test_data_root, Validation_data_root):
        self.test = self.get_Independent_image(Test_data_root)
        batch_idx, (data, target) = enumerate(self.test)
        print("\ntest_labels有" + str(len(self.test)) + "筆資料\n")

        self.validation = self.get_Independent_image(Validation_data_root)
        print("validation_labels有 " + str(len(self.validation)) + " 筆資料\n")

    def get_Independent_image(self, independent_DataRoot):
        read = Read_image_and_Process_image()

        Dataset = read.DataLoad_Image_Root(independent_DataRoot)
        # image_processing = Read_image_and_Process_image()

        # classify_image = []
        # Total_Dict_Data_Root = self.Get_Independent_data_Root(independent_DataRoot) # 讀取測試資料集的資料
        # Total_Dict_Data_Root = self.Specified_Amount_Of_Data(Total_Dict_Data_Root) # 打亂並取出指定資料筆數的資料
        # Total_List_Data_Root = [Total_Dict_Data_Root[self.Labels[0]], Total_Dict_Data_Root[self.Labels[1]]]            
        
        # test_label, Classify_Label = [], []
        # i = 0 # 計算classify_image的counter，且計算總共有幾筆資料
        # for test_title in Total_List_Data_Root: # 藉由讀取所有路徑來進行讀檔
        #     test_label = image_processing.make_label_list(len(test_title), self.OneHot_Encording[i]) # 製作對應圖片數量的label出來+    
        #     print(self.Labels[i] + " 有 " + str(len(test_label)) + " 筆資料 ")

        #     classify_image.append(test_title)
        #     Classify_Label.append(test_label)
        #     i += 1

        # original_test_root = self.merge.merge_data_main(classify_image, 0, 2)
        # original_test_label = self.merge.merge_data_main(Classify_Label, 0, 2)

        # test = []
        # test = image_processing.Data_Augmentation_Image(original_test_root)
        # test, test_label = image_processing.image_data_processing(test, original_test_label)
        # test = image_processing.normalization(test)

        return Dataset
    
    
    def Get_Independent_data_Root(self, load_data_root):
        Prepare = Load_Data_Prepare()
        Load_Tool = Load_Data_Tools()

        Prepare.Set_Data_Content([], len(self.Labels))
        Prepare.Set_Data_Dictionary(self.Labels, Prepare.Get_Data_Content(), 2)
        Get_Data_Dict_Content = Prepare.Get_Data_Dict()
        Total_Data_Roots = Load_Tool.get_data_root(load_data_root, Get_Data_Dict_Content, self.Labels)

        return Total_Data_Roots
    
    def Specified_Amount_Of_Data(self, Data): # 打亂資料後重新處理
        Data = shuffle_data(Data, self.Labels, 2)
        tmp = []
        if len(Data[self.Labels[0]]) >= len(Data[self.Labels[1]]):
            for i in range(len(Data[self.Labels[1]])):
                tmp.append(Data[self.Labels[0]][i])
            Data[self.Labels[0]] = tmp
        else:
            for i in range(len(Data[self.Labels[0]])):
                tmp.append(Data[self.Labels[1]][i])
            Data[self.Labels[1]] = tmp
        return Data