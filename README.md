main.py: 主程式檔

## load_process
### 負責讀取影像檔案、分割獨立資料(測試、驗證)、讀取獨立資料、一般檔案的操作
File_Process : 檔案操作的主程式，包含開檔、創立檔案、判斷檔案是否存在等都是他負責的範圍。是一般物件也是LoadData的父物件
LoadData : 讀檔主程式，一切讀檔動作由他開始。繼承File_Process(子物件)
Cutting_Indepentend_Image : 讀取獨立資料(testing、Validation)的物件

## Image_Process
### 負責進行資料擴增、影像處理等的操作
* Generator_Content : 負責建立基礎Generator項目，為Image_Generator的父類別
* Image_Generator : 負責製造資料擴增的資料，並將資料存到檔案中。繼承Generator_Content(子物件)
* image_enhancement : 負責進行影像處理並將資料回傳

## Model_Tools
### 負責進行模型的基礎架構，包含Convolution、Dense、以及其他模型的配件
* All_Model_Tools : 所有模型的附加工具，是所有的父類別

    ## CNN
    ### 包含所有CNN的工具與應用架構
    * CNN_Tools : 為卷積層的工具，包含一維、二維、三維捲積。CNN_Application的父類別，繼承All_Model_Tools(子類別)
    * CNN_Application : 為Convolution的應用架構。繼承CNN_Tools(子類別)

    ## Dense
    ### 包含所有Dense的應用
    * Dense_Application : 為全連階層工具，包含一般Dense layer與增加正則化之Dense layer。繼承All_Model_Tools()

    ## Model_Construction
    ### 包含所有要進行實驗的模型架構
    * Model_Constructions : 所有模型的實驗架構

## Data_Merge
### 負責進行資料的合併
* Merge : 負責合併Dict、List到List並匯出

## initalization
### 負責初始化特定物件
* Img_initalization : 針對影像資料的初始化
* Data_Initalization : 針對數據資料的初始化

## Validation_Program
### 負責驗證程式碼內的資料型態或輸入錯誤等問題
* Validation : 驗證程式碼錯誤

## Draw
### 負責畫圖的工具
* Draw_Tools : 畫出混淆矩陣、走勢圖的工具
* Grad_CAM : 畫出模型可視化的熱力圖的工具

## Experiment
### 執行實驗的主程式
* Experiment : 負責執行讀檔、設定模型Compile的細節、執行訓練、驗證結果等功能
