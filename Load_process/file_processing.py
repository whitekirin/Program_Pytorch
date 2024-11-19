import os
import cv2
import numpy as np
import datetime

class Process_File():
    def __init__(self) -> None:
        pass
    def JudgeRoot_MakeDir(self, file_root): # 先判斷檔案是否存在，再決定是否要進行開檔
        if self.Judge_File_Exist(file_root):
            return True
        else:
            self.Make_Dir(file_root)
            return False

    def Judge_File_Exist(self, file_root):
        '''判斷檔案是否存在，存在回傳true，否則為False'''
        if os.path.exists(file_root):
            return True 
        else:
            return False 
        
    def Make_Dir(self, file_root): # 建立資料夾
        os.makedirs(file_root)

    def Make_Save_Root(self, FileName, File_root): # 合併路徑
        return os.path.join(File_root, FileName)

    def Save_CV2_File(self, FileName, save_root, image): # 存CSV檔
        save_root = self.Make_Save_Root(FileName, save_root)
        cv2.imwrite(save_root, image)

    def Save_PIL_File(self, FileName, save_root, image): # 存CSV檔
        save_root = self.Make_Save_Root(FileName, save_root)
        image.save(save_root)

    def Save_NPY_File(self, FileName, save_root, image): # 存.npy檔
        save_root = self.Make_Save_Root(FileName, save_root)
        np.save(save_root, image)

    def Save_CSV_File(self, file_name, data): # 儲存訓練結果
        Save_Root = '../Result/save_the_train_result(' + str(datetime.date.today()) + ")"
        self.JudgeRoot_MakeDir(Save_Root)
        modelfiles = self.Make_Save_Root(file_name + ".csv", Save_Root)  # 將檔案名稱及路徑字串合併成完整路徑
        data.to_csv(modelfiles, mode = "a")

    def Save_TXT_File(self, content, File_Name):
        model_dir = '../Result/save_the_train_result(' + str(datetime.date.today()) + ")" # 儲存的檔案路徑，由save_the_train_result + 當天日期
        self.JudgeRoot_MakeDir(model_dir)
        modelfiles = self.Make_Save_Root(File_Name + ".txt", model_dir)  # 將檔案名稱及路徑字串合併成完整路徑
        with open(modelfiles, mode = 'a') as file:
            file.write(content)