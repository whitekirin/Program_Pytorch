import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Tool:
    def __init__(self) -> None:
        self.__ICG_Training_Root = ""
        self.__Normal_Training_Root = ""
        self.__Comprehensive_Training_Root = ""

        self.__ICG_Test_Data_Root = ""
        self.__Normal_Test_Data_Root = ""
        self.__Comprehensive_Testing_Root = ""

        self.__ICG_Validation_Data_Root = ""
        self.__Normal_Validation_Data_Root = ""
        self.__Comprehensive_Validation_Root = ""

        self.__ICG_ImageGenerator_Data_Root = ""
        self.__Normal_ImageGenerator_Data_Root = ""
        self.__Comprehensive_Generator_Root = ""

        self.__Labels = []
        self.__OneHot_Encording = []
        pass

    def Set_Labels(self):
        self.__Labels = ["stomach_cancer_Crop", "Normal_Crop"]

    def Set_Save_Roots(self):
        self.__ICG_Training_Root = "../Dataset/Training/CA_ICG"
        self.__Normal_Training_Root = "../Dataset/Training/CA"
        self.__Comprehensive_Training_Root = "../Dataset/Training/Mixed"

        self.__ICG_Test_Data_Root = "../Dataset/Training/CA_ICG_TestData"
        self.__Normal_Test_Data_Root = "../Dataset/Training/Normal_TestData"
        self.__Comprehensive_Testing_Root = "../Dataset/Training/Comprehensive_TestData"

        self.__ICG_Validation_Data_Root = "../Dataset/Training/CA_ICG_ValidationData"
        self.__Normal_Validation_Data_Root = "../Dataset/Training/Normal_ValidationData"
        self.__Comprehensive_Validation_Root = "../Dataset/Training/Comprehensive_ValidationData"

        self.__ICG_ImageGenerator_Data_Root = "../Dataset/Training/ICG_ImageGenerator"
        self.__Normal_ImageGenerator_Data_Root = "../Dataset/Training/Normal_ImageGenerator"
        self.__Comprehensive_Generator_Root = "../Dataset/Training/Comprehensive_ImageGenerator"

    def Set_OneHotEncording(self, Content):
        Array_To_DataFrame = pd.DataFrame(Content)
        onehotencoder = OneHotEncoder()
        onehot = onehotencoder.fit_transform(Array_To_DataFrame).toarray()
        
        self.__OneHot_Encording = onehot

    def Get_Data_Label(self):
        '''
        取得所需資料的Labels
        '''
        return self.__Labels
    
    def Get_Save_Roots(self, choose):
        '''回傳結果為Train, test, validation
                choose = 1 => 取ICG Label
                else => 取Normal Label

            若choose != 1 || choose != 2 => 會回傳四個結果
        '''
        if choose == 1:
            return self.__ICG_Training_Root, self.__ICG_Test_Data_Root, self.__ICG_Validation_Data_Root
        if choose == 2:
            return self.__Normal_Training_Root, self.__Normal_Test_Data_Root, self.__Normal_Validation_Data_Root
        else:
            return self.__Comprehensive_Training_Root, self.__Comprehensive_Testing_Root, self.__Comprehensive_Validation_Root
        
    def Get_Generator_Save_Roots(self, choose):
        '''回傳結果為Train, test, validation'''
        if choose == 1:
            return self.__ICG_ImageGenerator_Data_Root
        if choose == 2:
            return self.__Normal_ImageGenerator_Data_Root
        else:
            return self.__Comprehensive_Generator_Root
        
    def Get_OneHot_Encording_Label(self):
        return self.__OneHot_Encording