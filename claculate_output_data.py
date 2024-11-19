import csv
import numpy as np

judge = input("是否需要移動? (Y/N)")

if judge == 'y' or judge == 'Y':
    times = int(input("輸入要移動幾天: "))

    for i in range(times):
        date = input("輸入被移動的日期: ")
        dateroot = "../Model_training_result/save_the_train_result(2024-" + date + ")/train_result.csv"
        quantity_data = int(input("輸入要取出的資料筆數: "))

        next_date = input("移動到哪一天? ")

        with open(dateroot, "r", newline = '') as csvFile:
            data = csv.reader(csvFile)
            data = list(data)

            with open("../Model_training_result/save_the_train_result(2024-" + next_date + ")/train_result.csv", "a+", newline = '') as csvFile1:
                writer = csv.writer(csvFile1)
                for i in range((quantity_data * 2 + 1) * -1 + 1, 0, 1):
                    writer.writerow(data[i])
                print("Data has been moved finish\n")



date = input("輸入計算的日期: ")
with open("../Model_training_result/save_the_train_result(2024-" + date + ")/train_result.csv", newline = '') as csvfile:
    rows = csv.reader(csvfile)

    row = list(rows)

    calcalate_loss = 0
    calculate_precision = 0
    calculate_recall = 0
    calculate_accuracy = 0
    calculate_f = 0
    calculate_auc = 0

    list_loss = []
    list_precision = []
    list_recall = []
    list_accuracy = []
    list_f = []
    list_auc = []
    
    for i in range(-1, -10, -2):
        calcalate_loss += float(row[i][2])
        list_loss.append(float(row[i][2]))

        precision = str(row[i][3]).split("%")
        calculate_precision += float(precision[0])
        list_precision.append(float(precision[0]))
        

        recall = str(row[i][4]).split("%")
        calculate_recall += float(recall[0])
        list_recall.append(float(recall[0]))

        accuracy = str(row[i][5]).split("%")
        calculate_accuracy += float(accuracy[0])
        list_accuracy.append(float(accuracy[0]))

        f = str(row[i][6]).split("%")
        calculate_f += float(f[0])
        list_f.append(float(f[0]))

        auc = str(row[i][7]).split("%")
        calculate_auc += float(auc[0])
        list_auc.append(float(auc[0]))

calculate_list = [calcalate_loss, calculate_precision, calculate_recall, calculate_accuracy, calculate_f, calculate_auc]
average = []
for i in range(len(calculate_list)):
    average.append((calculate_list[i] / 5))

std_list = [list_precision, list_recall, list_accuracy, list_f, list_auc]
standard = []
standard.append(np.std(list_loss))
for i in range(len(std_list)):
    standard.append((np.std(std_list[i]) / 100))

for i in range(len(average)):
    print("{:.2f}±{:.3f}".format(average[i], standard[i]))