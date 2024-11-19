import random


def calculate_confusion_matrix(predict, result):
    '''計算並畫出混淆矩陣'''
    tp, fp, tn, fn = 0
    for i in range(len(predict)):
        if predict[i] == [1., 0., 0.] and result[i] == [1., 0., 0.]:
           pass 


def shuffle_data(image, label, mode = 1):
    '''
    ## 被用來做資料打亂的用途
    ### 有兩種不同的需求
    1. 打亂影像資料(讀完檔後的影像) => 回傳Label與Image Root兩個List
    2. 打亂路徑資料(影像的路徑資料，還沒讀檔前) => 回傳打亂後的Dict
    '''
    if mode == 1:
        shuffle_image, shuffle_label = [], []

        total = list(zip(image, label))
        random.shuffle(total)

        for total_data in total:
            shuffle_image.append(total_data[0])
            shuffle_label.append(total_data[1])

        return shuffle_image, shuffle_label
    else:
        shuffle_image = {
        label[0] : [],
        label[1] : [],
        }
        for Label in label:
            shuffle_image[Label] = image[Label]
            random.shuffle(shuffle_image[Label])

        return shuffle_image