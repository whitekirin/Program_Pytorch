from Load_process.file_processing import Process_File
from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image
from matplotlib import pyplot as plt
import torch
import cv2
import numpy as np
import datetime

class Grad_CAM:
    def __init__(self, Label, One_Hot, Experiment_Name, Layer_Name) -> None:
        self.experiment_name = Experiment_Name
        self.Layer_Name = Layer_Name
        self.Label = Label
        self.One_Hot_Label = One_Hot
        self.Save_File_Name = self.Convert_One_Hot_To_int()

        pass

    def process_main(self, model, index, images):
        cam_extractor = GradCAM(model, target_layer=self.Layer_Name)

        for i, image in enumerate(images):
            input_tensor = image.unsqueeze(0)  # 在 PyTorch 中增加批次維度
            heatmap = self.gradcam(input_tensor, model, cam_extractor)
            self.plot_heatmap(heatmap, image, self.Save_File_Name[i], index, i)
        pass

    def Convert_One_Hot_To_int(self):
        return [np.argmax(Label)for Label in self.One_Hot_Label]


    def gradcam(self, Image, model, cam_extractor):
        # 將模型設為評估模式
        model.eval()
        # 前向傳播並生成熱力圖
        with torch.no_grad():
            out = model(Image)

        # 取得預測類別
        pred_index = out.argmax(dim=1).item()
        
        # 生成對應的 Grad-CAM 熱力圖
        heatmap = cam_extractor(pred_index, out)
        return heatmap[0].cpu().numpy()
    
    def plot_heatmap(self, heatmap, img, Label, index, Title):
        File = Process_File()
        
        # 調整影像大小
        img_path = cv2.resize(img.numpy().transpose(1, 2, 0), (512, 512))
        heatmap = cv2.resize(heatmap, (512, 512))
        heatmap = np.uint8(255 * heatmap)

        img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
        
        # 顯示影像和熱力圖
        fig, ax = plt.subplots()
        ax.imshow(img_path, alpha=1)
        ax.imshow(heatmap, cmap='jet', alpha=0.3)

        save_root = '../Result/CNN_result_of_reading('+ str(datetime.date.today()) + ")/" + str(Label)
        File.JudgeRoot_MakeDir(save_root)
        save_root = File.Make_Save_Root(self.experiment_name + "-" + str(index) + "-" + str(Title) + ".png", save_root)
        
        plt.savefig(save_root)
        plt.close("all")
        pass