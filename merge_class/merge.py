from _validation.ValidationTheEnterData import validation_the_enter_data

class merge:
    def __init__(self) -> None:
        self.validation = validation_the_enter_data()
        pass

    def set_merge_data(self, merge_data):
        self.__merge_data = merge_data
        pass

    def set_merge_keys(self, dict_key):
        self.__keys = dict_key

    def set_judge_status(self, judge):
        self.__judge = judge

    def get_judge_status(self):
        return self.__judge
    
    def merge_all_image_data(self, Classify1, Classify2):
        merged_data = [Classify1, Classify2]

        return self.merge_data_main(merged_data, 0, 2)

    def merge_data_main(self, merge_data, merge_start_index, total_merge_number = 3):
        '''
        將各類別資料合併在一起
        ## Parameter:
            * merge_data: 要被合併的資料
            * merge_start_index: 合併資料的起始位置
            * total_merge_numbers: 總共要合併的數量
        '''
        if self.validation.validation_type(merge_data, dict):
            self.set_merge_data(merge_data)
            return self.merge_dict_to_list_data(merge_start_index, total_merge_number)
            
        elif self.validation.validation_type(merge_data, list):
            self.set_merge_data(merge_data)
            return self.merge_list_to_list_data(merge_start_index, total_merge_number)
        
    def merge_dict_to_list_data(self, merge_start_index, total_merge_number = 3):
        self.set_merge_keys(list(self.__merge_data.keys()))

        self.set_judge_status(1)
        result = list(self.__merge_data[self.__keys[merge_start_index]])
        result = self.merge_loop(result, merge_start_index, total_merge_number)
        
        return result
    
    def merge_dict_to_dict(self, original : dict, myself):
        keys = list(original.keys())
        data = {
            keys[0]: [],
            keys[1]: [],
        }
        
        for key in keys:
            tempData = [original[key], myself[key]]
            end = 2

            self.set_merge_data(tempData)
            data[key] = self.merge_list_to_list_data(0, end)

        return data
        
    def merge_list_to_list_data(self, merge_start_index, total_merge_number = 3):
        self.set_judge_status(2)
        result = list(self.__merge_data[merge_start_index])
        return self.merge_loop(result, merge_start_index, total_merge_number)

    def merge_loop(self, result, merge_start_index, total_merge_number = 3):
        for i in range(merge_start_index + 1, merge_start_index + total_merge_number, 1):
            if self.get_judge_status() == 1:
                result += list(self.__merge_data[self.__keys[i]])
            else:
                result += list(self.__merge_data[i])

        return result