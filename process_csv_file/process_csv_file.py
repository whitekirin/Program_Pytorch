import csv

def get_csv_file_data(read_filename):
    with open("../../Model_training_result/save_the_train_result(" + read_filename + ")/train_result.csv", newline = '') as csvfile:
        rows = csv.reader(csvfile)

        row = list(rows)
    return row