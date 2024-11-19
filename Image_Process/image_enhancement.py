import cv2
import numpy as np

def shapen(image): # 銳化處理
    sigma = 100
    blur_img = cv2.GaussianBlur(image, (0, 0), sigma)
    usm = cv2.addWeighted(image, 1.5, blur_img, -0.5, 0)

    return usm

def increase_contrast(image): # 增加資料對比度
    output = image    # 建立 output 變數
    alpha = 2
    beta = 10
    cv2.convertScaleAbs(image, output, alpha, beta)  # 套用 convertScaleAbs

    return output

def adaptive_histogram_equalization(image):    
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])

    ycrcb = cv2.merge(channels)
    Change_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    
    return Change_image

def Remove_Background(image, Matrix_Size):    
    skinCrCbHist = np.zeros((256,256), dtype= np.uint8)
    cv2.ellipse(skinCrCbHist, (113,155),(23,25), 43, 0, 360, (255, 255, 255), -1) #繪製橢圓弧線

    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    y,cr,cb = cv2.split(img_ycrcb) #拆分出Y,Cr,Cb值

    skin = np.zeros(cr.shape, dtype = np.uint8) #掩膜
    (x,y) = cr.shape

    # 依序取出圖片中每個像素
    for i in range(x):
        for j in range(y):
            if skinCrCbHist [cr[i][j], cb[i][j]] > 0: #若不在橢圓區間中
                skin[i][j] = 255
                    # 如果該像素的灰階度大於 200，調整該像素的透明度
                    # 使用 255 - gray[y, x] 可以將一些邊緣的像素變成半透明，避免太過鋸齒的邊緣
    # img_change = cv2.cvtColor(img_change, cv2.COLOR_BGRA2BGR)
    img = cv2.bitwise_and(image, image, mask = skin)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h = image.shape[0]     # 取得圖片高度
    w = image.shape[1]     # 取得圖片寬度

    for x in range(w):
        for y in range(h):
            if img_gray[y, x] == 0:
                # if x == 0 and y == 0:  # 當X Y都在左上角時
                #     image[y, x] = Add(1, Matrix_Size, image[y, x]) / Matrix_Size
                # if x == w - 1 and y == 0: # 當X Y都在右上角時
                #     image[y, x] = Add(w - Matrix_Size, w, image[y, x]) / Matrix_Size
                # if x == 0 and y == h - 1: # 當X Y都在左下角時
                #     image[y, x] = (image[y - 1, x] + image[y - 1, x + 1] + image[y, x + 1]) / 3
                # if x == w - 1 and y == h - 1: # 當X Y都在右下角時
                #     image[y, x] = (image[y, x - 1] + image[y - 1, x - 1] + image[y - 1, x]) / 3

                # if (x > 0 and x < w - 1) and y == 0: # 當上面的X Y從左到右
                #     image[y, x] = (image[y, x - 1] + image[y + 1, x - 1] + image[y + 1, x] + image[y, x + 1] + image[y + 1, x + 1]) / 5
                # if (x > 0 and x < w - 1) and y == h - 1: # 當下面的X Y從左到右
                #     image[y, x] = (image[y, x - 1] + image[y - 1, x - 1] + image[y - 1, x] + image[y, x + 1] + image[y - 1, x + 1]) / 5
                # if x == 0 and (y > 0 and y < h - 1): # 當左邊的X Y從上到下
                #     image[y, x] = (image[y - 1, x] + image[y - 1, x + 1] + image[y, x + 1] + image[y + 1, x + 1] + image[y + 1, x]) / 5
                # if x == w - 1 and (y > 0 and y < h - 1): # 當右邊X Y從上到下
                #     image[y, x] = (image[y - 1, x] + image[y - 1, x - 1] + image[y, x - 1] + image[y + 1, x - 1] + image[y + 1, x]) / 5

                if (x >= 1 and x < w - 1) and (y >= 1 and y < h - 1): # 當y >= 2 且 X >= 2
                    image[y, x] = Add(x, y, image, Matrix_Size) / Matrix_Size
                # BGRA_image[y, x, 3] = 255 - gray[y, x]
    return image


def Add(width_Center, Height_Center, image, Mask_Size):
    total = 0
    for i in range(Mask_Size):
        for j in range(Mask_Size):
            total += image[width_Center - ((Mask_Size - 1) / 2) + j, Height_Center - ((Mask_Size - 1) / 2) + i]

    return total