from _validation.ValidationTheEnterData import validation_the_enter_data
from Load_process.file_processing import Process_File
from torchvision import transforms
from Load_process.LoadData import Load_Data_Prepare, Load_Data_Tools
from torch.utils.data import Dataset

class Image_generator():
    '''製作資料強化'''
    def __init__(self, Generator_Root, Labels) -> None:
        self._validation = validation_the_enter_data()
        self.stop = 0
        self.Labels = Labels
        self.Generator_Root = Generator_Root        
        pass

    def Processing_Main(self, Training_Dict_Data_Root):
        data_size = 0

        # 製作標準資料增強
        '''
            這裡我想要做的是依照paper上的資料強化IMAGE DATA COLLECTION AND IMPLEMENTATION OF DEEP LEARNING-BASED MODEL IN DETECTING MONKEYPOX DISEASE USING MODIFIED VGG16
            產生出資料強化後的影像
        '''
        print("\nAugmentation one monkeypox image")
        data_size = self.get_processing_Augmentation(Training_Dict_Data_Root, 1, data_size)
        self.stop += data_size

        # 製作標準資料增強
        '''
            這裡我想要做的是依照paper上的資料強化IMAGE DATA COLLECTION AND IMPLEMENTATION OF DEEP LEARNING-BASED MODEL IN DETECTING MONKEYPOX DISEASE USING MODIFIED VGG16
            產生出資料強化後的影像
        '''
        print("\nAugmentation two monkeypox image")
        data_size = self.get_processing_Augmentation(Training_Dict_Data_Root, 2, data_size)
        self.stop += data_size

        # 製作標準資料增強
        '''
            這裡我想要做的是依照paper上的資料強化IMAGE DATA COLLECTION AND IMPLEMENTATION OF DEEP LEARNING-BASED MODEL IN DETECTING MONKEYPOX DISEASE USING MODIFIED VGG16
            產生出資料強化後的影像
        '''
        print("\nAugmentation three monkeypox image")
        data_size = self.get_processing_Augmentation(Training_Dict_Data_Root, 3, data_size)
        self.stop += data_size
        

        # 製作標準資料增強
        '''
            這裡我想要做的是依照paper上的資料強化IMAGE DATA COLLECTION AND IMPLEMENTATION OF DEEP LEARNING-BASED MODEL IN DETECTING MONKEYPOX DISEASE USING MODIFIED VGG16
            產生出資料強化後的影像
        '''
        print("\nAugmentation four monkeypox image")
        data_size = self.get_processing_Augmentation(Training_Dict_Data_Root, 4, data_size)

        print()

    def get_processing_Augmentation(self, original_image_root : dict, Augment_choose, data_size):
        Prepaer = Load_Data_Prepare()

        self.get_data_roots = original_image_root # 要處理的影像路徑
        Prepaer.Set_Label_List(self.Labels)
        data_size = self.Generator_main(self.Generator_Root, Augment_choose, data_size) # 執行
        return data_size
    
    def Generator_main(self, save_roots, stardand, data_size):
        '''
            Parameter:
                labels = 取得資料的標籤
                save_root = 要儲存資料的地方
                strardand = 要使用哪種Image Augmentation
        '''
        File = Process_File()

        for label in self.Labels: # 分別對所有類別進行資料強化
            image = self.load_data(stardand) # 取的資料
            save_root = File.Make_Save_Root(label, save_roots) # 合併路徑

            if File.JudgeRoot_MakeDir(save_root): # 判斷要存的資料夾存不存在，不存在則創立
                print("The file is exist")

            stop_counter = 0
            for batch_idx, (images, labels) in enumerate(image):
                for i, img in enumerate(images):
                    img_pil = transforms.ToPILImage()(img)
                    File.Save_PIL_File("image_" + label + str(data_size) + ".png", save_root, img_pil) # 存檔

        return data_size

    def load_data(self, judge):
        '''Images is readed by myself'''
        Load_Tools = Load_Data_Tools()
        image_load = Load_Tools.Load_ImageFolder_Data(self.get_data_roots, self.Generator_Content(judge))
        img = Load_Tools.DataLoad_Image_Root(image_load, 16)
        # img = image_processing.Data_Augmentation_Image(self.get_data_roots[label])

        self.stop = len(img) * 1.5
        return img
    
    def Generator_Content(self, judge): # 影像資料增強
        '''
        ImageGenerator的參數:
            featurewise_center : 布爾值。將輸入數據的均值設置為0，逐特徵進行。
            samplewise_center : 布爾值。將每個樣本的均值設置為0。
            featurewise_std_normalization : Boolean. 布爾值。將輸入除以數據標準差，逐特徵進行。
            samplewise_std_normalization : 布爾值。將每個輸入除以其標準差。
            zca_epsilon : ZCA 白化的epsilon 值，默認為1e-6。
            zca_whitening : 布爾值。是否應用ZCA 白化。
            rotation_range : 整數。隨機旋轉的度數範圍。
            width_shift_range : 浮點數、一維數組或整數
                float: 如果<1，則是除以總寬度的值，或者如果>=1，則為像素值。
                1-D 數組: 數組中的隨機元素。
                int: 來自間隔 (-width_shift_range, +width_shift_range) 之間的整數個像素。
                width_shift_range=2時，可能值是整數[-1, 0, +1]，與 width_shift_range=[-1, 0, +1] 相同；而 width_shift_range=1.0 時，可能值是 [-1.0, +1.0) 之間的浮點數。
            height_shift_range : 浮點數、一維數組或整數
                float: 如果<1，則是除以總寬度的值，或者如果>=1，則為像素值。
                1-D array-like: 數組中的隨機元素。
                int: 來自間隔 (-height_shift_range, +height_shift_range) 之間的整數個像素。
                height_shift_range=2時，可能值是整數[-1, 0, +1]，與 height_shift_range=[-1, 0, +1] 相同；而 height_shift_range=1.0 時，可能值是 [-1.0, +1.0) 之間的浮點數。
            shear_range : 浮點數。剪切強度（以弧度逆時針方向剪切角度）。
            zoom_range : 浮點數或[lower, upper]。隨機縮放範圍。如果是浮點數，[lower, upper] = [1-zoom_range, 1+zoom_range]。
            channel_shift_range : 浮點數。隨機通道轉換的範圍。
            fill_mode : {"constant", "nearest", "reflect" or "wrap"} 之一。默認為'nearest'。輸入邊界以外的點根據給定的模式填充：
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest': aaaaaaaa|abcd|dddddddd
                'reflect': abcddcba|abcd|dcbaabcd
                'wrap': abcdabcd|abcd|abcdabcd
            cval : 浮點數或整數。用於邊界之外的點的值，當 fill_mode = "constant" 時。
            horizontal_flip : 布爾值。隨機水平翻轉。
            vertical_flip : 布爾值。隨機垂直翻轉。
            rescale : 重縮放因子。默認為None。如果是None 或0，不進行縮放，否則將數據乘以所提供的值（在應用任何其他轉換之前）。
            preprocessing_function : 應用於每個輸入的函數。這個函數會在任何其他改變之前運行。這個函數需要一個參數：一張圖像（秩為3 的Numpy 張量），並且應該輸出一個同尺寸的Numpy 張量。
            data_format : 圖像數據格式，{"channels_first", "channels_last"} 之一。"channels_last" 模式表示圖像輸入尺寸應該為(samples, height, width, channels)，"channels_first" 模式表示輸入尺寸應該為(samples, channels, height, width)。默認為在Keras 配置文件 ~/.keras/keras.json 中的 image_data_format 值。如果你從未設置它，那它就是"channels_last"。
            validation_split : 浮點數。Float. 保留用於驗證的圖像的比例（嚴格在0和1之間）。
            dtype : 生成數組使用的數據類型。
        '''
        if judge == 1:
            return transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        elif judge == 2:
            return transforms.Compose([
                transforms.RandomRotation(180),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        elif judge == 3:
            return transforms.Compose([
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.RandomAffine(degrees=20, shear=0.2),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomHorizontalFlip(),
            ])
        elif judge == 4:
            return transforms.Compose([
                transforms.RandomRotation(50),
                transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                transforms.RandomAffine(degrees=30, shear=0.25),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:
            return transforms.ToTensor() # 將數值歸一化到[0, 1]之間