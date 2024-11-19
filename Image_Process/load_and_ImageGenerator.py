from Load_process.LoadData import Loding_Data_Root
from Image_Process.Image_Generator import Image_generator
from Load_process.file_processing import Process_File
from model_data_processing.processing_for_cut_image import Cut_Indepentend_Data
from Load_process.Loading_Tools import Load_Data_Prepare, Load_Data_Tools

class Load_ImageGenerator():
    '''
    這是一個拿來進行資料強化的物件，最主要結合了學姊給的資料強化與我自行設定的資料強化。
藉由此物件先將資料讀取出來，並將資料分別進行資料強化，利用資料強化來迷部資料的不平衡
這只是其中一個實驗

Parmeter
    standard_root: 做跟學姊給的資料強化同一種的資料強化
    myself_root: 資料強化的內容參數是我自己設定的
    IndependentDataRoot: 要存回去的資料夾路徑
    Herpeslabels: 皰疹的類別
    MonKeyPoxlabels: 猴痘的類別(猴痘、水痘、正常)
    herpes_data: 合併herpes Dataset的資料成一個List
    MonkeyPox_data: 合併MonkeyPox DataSet 的資料成一個List
    '''
    def __init__(self, Training_Root,Test_Root, Validation_Root, Generator_Root, Labels) -> None:
        self.Training_Root = Training_Root
        self.TestRoot = Test_Root
        self.ValidationRoot = Validation_Root
        self.GeneratoRoot = Generator_Root
        self.Labels = Labels
        pass

    def process_main(self, Data_Length : int):
        File = Process_File()
        Prepare = Load_Data_Prepare()
        load = Loding_Data_Root(self.Labels, self.Training_Root, self.GeneratoRoot)
        Indepentend = Cut_Indepentend_Data(self.Training_Root, self.Labels)
        Load_Tool = Load_Data_Tools()
        Generator = Image_generator(self.GeneratoRoot, self.Labels)

        # 將測試資料獨立出來
        test_size = 0.1
        Indepentend.IndependentData_main(self.TestRoot, test_size)

        # 將驗證資料獨立出來
        test_size = 0.1
        Indepentend.IndependentData_main(self.ValidationRoot, test_size)

        if not File.Judge_File_Exist(self.GeneratoRoot): # 檔案若不存在
            # 確定我要多少個List
            Prepare.Set_Data_Content([], Data_Length)

            # 製作讀檔字典並回傳檔案路徑
            Prepare.Set_Label_List(self.Labels)
            Prepare.Set_Data_Dictionary(Prepare.Get_Label_List(), Prepare.Get_Data_Content(), Data_Length)
            Original_Dict_Data_Root = Prepare.Get_Data_Dict()
            get_all_original_image_data = Load_Tool.get_data_root(self.Training_Root, Original_Dict_Data_Root, Prepare.Get_Label_List())

            # 儲存資料強化後資料
            Generator.Processing_Main(get_all_original_image_data) # 執行資料強化
        else: # 若檔案存在
            print("standard data and myself data are exist\n")
        
        # 執行讀檔
        return load.process_main()