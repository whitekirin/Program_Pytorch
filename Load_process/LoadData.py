from Load_process.file_processing import Process_File
from Load_process.Loading_Tools import Load_Data_Prepare, Load_Data_Tools
from merge_class.merge import merge
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import ConcatDataset

class Loding_Data_Root(Process_File):
    def __init__(self, Labels, Training_Root, Generator_Root):
        self.Label_List = Labels
        self.Train_Root = Training_Root
        self.Generator_Root = Generator_Root

        super().__init__()
        pass

    def process_main(self):
        '''處理讀Training、Image Generator檔資料'''
        # Merge = merge()
        Loading_Tool = Load_Data_Tools()

        # 在後面加上transform的原因是要讓讀進來的內容轉成tensor的格式
        get_Image_Data = Loading_Tool.Load_ImageFolder_Data(self.Train_Root, "transform")
        Get_ImageGenerator_Image_Data = Loading_Tool.Load_ImageFolder_Data(self.Generator_Root, "transform")

        Total_Data_List = get_Image_Data.targets + Get_ImageGenerator_Image_Data.targets
        Get_Total_Image_Data_Root = ConcatDataset([get_Image_Data, Get_ImageGenerator_Image_Data])

        return Get_Total_Image_Data_Root, Total_Data_List