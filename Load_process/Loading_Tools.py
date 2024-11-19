import os
import glob
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np

class Load_Data_Prepare:
    def __init__(self) -> None:
        self.__Label_List = []
        self.__Data_List = []
        self.__Contect_Dictionary = {}
        self.__Final_Dict_data = {}
        self.__PreSave_Data_Root = [] # 所有要讀取資料所在的位置
        self.__Data_Content = []
        pass

    def Set_Data_Content(self, Content, Length):
        tmp = []
        for i in range(Length):
            tmp.append(Content)

        self.__Data_Content = tmp

    def Set_Label_List(self, Label_List): # 為讀取檔案準備label list
        self.__Label_List = Label_List
        pass

    def Set_Data_List(self, Data_List):
        self.__Data_List = Data_List
        pass

    def Set_Data_Dictionary(self, Label : list, Content : list, Total_Label_Size : int):
        '''將資料合併成1個Dict'''
        for i in range(Total_Label_Size):
            temp = {Label[i] : Content[i]}
            self.__Contect_Dictionary.update(temp)
        pass

    def Set_Final_Dict_Data(self, Name : list, Label_Root : list, Label_LabelEncoding : list, Label_Len : int):
        '''
        Name : 讀取出來的Data Root的名字
        Label_Root: 所有影像資料的路徑
        Label_LabelEncoding: LabelEncoding後的資料
        Label_Len: Label的大小
        '''
        for i in range(Label_Len):
            temp = {Name[i] + "_Data_Root" : Label_Root[Name[i]]}
            self.__Final_Dict_data.update(temp)

        for i in range(Label_Len):
            temp = {Name[i] + "_Data_LabelEncoding" : Label_LabelEncoding[i]}
            self.__Final_Dict_data.update(temp)

    def Set_PreSave_Data_Root(self, PreSave_Roots : list):
        for Root in PreSave_Roots:
            self.__PreSave_Data_Root.append(Root)

    def Get_Label_List(self): 
        ''' 
        將private的資料讀取出來
        現在要放入需要的Label 需要先Set Label
        '''
        return self.__Label_List
    
    def Get_Data_List(self):
        return self.__Data_List
    
    def Get_Data_Dict(self):
        return self.__Contect_Dictionary
    
    def Get_Final_Data_Dict(self):
        return self.__Final_Dict_data
    
    def Get_PreSave_Data_Root(self):
        return self.__PreSave_Data_Root
    
    def Get_Data_Content(self):
        return self.__Data_Content

class Load_Data_Tools():
    def __init__(self) -> None:
        pass

    def get_data_root(self, root, data_dict, classify_label, judge = True) -> dict :
        '''取得資料路徑'''
        for label in classify_label:
            if judge:
                path = os.path.join(root, label, "*")
            else:
                path = os.path.join(root, "*")
            path = glob.glob(path)
            data_dict[label] = path
        return data_dict
    
    def Load_ImageFolder_Data(self, Loading_Root, transform = None):
        # 資料預處理
        if transform == None:
            transform = transforms.Compose([
                transforms.Resize((512, 512))
            ])
        else:
             transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])

        DataSet = ImageFolder(root=Loading_Root, transform=transform)

        return DataSet
    
    def Get_Balance_Data(self, Image_Folder : ImageFolder, total_Image_List):
        class_counts = np.bincount(total_Image_List)
        print("欄位大小: " + str(total_Image_List))
        min_class_count = class_counts.min()

        # 創建每個類別的索引並下採樣
        balanced_indices = []
        for class_idx in range(len(class_counts)):
            class_indices = np.where(np.array(total_Image_List) == class_idx)[0]
            sampled_indices = np.random.choice(class_indices, min_class_count, replace=False)
            balanced_indices.extend(sampled_indices)

        # 創建平衡的子集
        Training_Data = Subset(Image_Folder, balanced_indices)

        # 輸出內容
        print(f"平衡後的每類樣本數：{min_class_count}")

        return Training_Data
    
    def DataLoad_Image_Root(self, ImageLoad, Batch):
        dataloader = DataLoader(dataset = ImageLoad, batch_size = Batch, shuffle=True, num_workers = 0, pin_memory=True)
        return dataloader