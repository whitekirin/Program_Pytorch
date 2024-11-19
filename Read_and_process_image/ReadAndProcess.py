import cv2
import numpy as np


class Read_image_and_Process_image:
    def __init__(self) -> None:
        pass
    def get_data(self, path):
        '''讀檔'''
        img_size = 512 # 縮小後的影像
        try:
            img_arr = cv2.imread(path, cv2.IMREAD_COLOR) # 讀檔(彩色)
            # img_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 讀檔(灰階)
            resized_arr = cv2.resize(img_arr, (img_size, img_size)) # 濤整圖片大小
        except Exception as e:
            print(e)
        
        return resized_arr
    
    def Data_Augmentation_Image(self, path):
        resized_arr = []

        for p in path:
            img_size = 512 # 縮小後的影像
            try:
                img_arr = cv2.imread(p, cv2.IMREAD_COLOR) # 讀檔(彩色)
                # img_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 讀檔(灰階)
                resized_arr.append(cv2.resize(img_arr, (img_size, img_size))) # 濤整圖片大小
            except Exception as e:
                print(e)
        
        return np.array(resized_arr)

    def image_data_processing(self, data, label):
        '''讀檔後處理圖片'''
        img_size = 512
        data = np.asarray(data).astype(np.float32) # 將圖list轉成np.array
        data = data.reshape(-1, img_size, img_size, 3)  # 更改陣列形狀
        label = np.array(label)                         # 將label從list型態轉成 numpy array
        return data, label
    
    def normalization(self, images):
        imgs = []
        for img in images:
            img = np.asarray(img).astype(np.float32) # 將圖list轉成np.array
            img = img / 255                     # 標準化影像資料
            imgs.append(img)

        return np.array(imgs)
    
    # def load_numpy_data(self, file_names):
    #     '''載入numpy圖檔，並執行影像處理提高特徵擷取'''
    #     i = 0
    #     numpy_image = []
    #     original_image = []
    #     for file_name in file_names:
    #         compare = str(file_name).split(".")
    #         if compare[-1] == "npy":
    #             image = np.load(file_name) # 讀圖片檔
    #             numpy_image.append(image) # 合併成一個陣列
    #         else:
    #             original_image.append(file_name)

    #     original_image = self.get_data(original_image)

    #     for file in original_image:
    #         numpy_image.append(file)
            
    #     return numpy_image

    def make_label_list(self, length, content):
        '''製作label的列表'''
        label_list = []
        for i in range(length):
            label_list.append(content)
        return label_list