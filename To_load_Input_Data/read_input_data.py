import tensorflow as tf
import os

class read_Input_Data:
    def __init__(self) -> None:
        pass
    def save_tfrecords(self, images, label):
        '''將資料儲存為TFRecord數據'''
        image_width, image_height = 64, 64
        image_channel = 3
        tfrecod_data_root = "../../Dataset/tfrecode_Dataset/tfrecod_data.tfrecords"
        if not os.path.exists(tfrecod_data_root):
            os.makedirs(tfrecod_data_root)

        TFWriter = tf.python_io.TFRecordWriter(tfrecod_data_root)

        try:
            for i in range(len(images)):
                if images[i] is None:
                    print('Error image:' + images[i])
                else:
                    #圖片轉為字串
                    image_raw = str(images[i])
    

                 # 將 tf.train.Feature 合併成 tf.train.Features
                train_feature = tf.train.Features(feature={
                    'Label' : self.int64_feature(label),
                    'image_raw' : self.bytes_feature(image_raw),
                    'channel' : self.int64_feature(image_channel),
                    'width' : self.int64_feature(image_width),
                    'height' : self.int64_feature(image_height)}
                   )

                # 將 tf.train.Features 轉成 tf.train.Example
                train_example = tf.train.Example(features = train_feature)

                # 將 tf.train.Example 寫成 tfRecord 格式
                TFWriter.write(train_example.SerializeToString())
                    
        except Exception as e:
            print(e)

        TFWriter.close()
        print('Transform done!')

        return tfrecod_data_root
        
    # 轉Bytes資料為 tf.train.Feature 格式
    def int64_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def Data_Decompile(self, example):
        '''反編譯TFR數據'''
        feature_description = {
            'data': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.float32),
        }
        parsed_example = tf.io.parse_single_example(example, features=feature_description)
        
        x_sample = tf.io.parse_tensor(parsed_example['data'], tf.float32)
        y_sample = parsed_example['label']
        
        return x_sample, y_sample

    def load_dataset(self, filepaths):
        '''
        載入TFR數據集
        * dataset.shuffle(shuffle_buffer_size):
            隨機打亂此數據集的元素。

            該數據集用 buffer_size 元素填充緩衝區，然後從該緩衝區中隨機採樣元素，用新元素替換所選元素。
            為了完美改組，需要緩衝區大小大於或等於數據集的完整大小。

            例如，如果您的數據集包含 10,000 個元素，但 buffer_size 設置為 1,000,
            則 shuffle 最初只會從緩衝區的前 1,000 個元素中選擇一個隨機元素。 
            一旦選擇了一個元素，它在緩衝區中的空間將被下一個（即第 1,001 個）元素替換，從而保持 1,000 個元素的緩衝區。
        '''
        shuffle_buffer_size = 700
        batch_size = 128
        tfrecod_data_root = "../../Dataset/tfrecode_Dataset"

        dataset = tf.data.TFRecordDataset(filepaths)
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.map(map_func=self.Data_Decompile, num_parallel_calls= 8)
        dataset = dataset.batch(batch_size).prefetch(64)

        # 產生文件名隊列
        filename_queue = tf.train.string_input_producer([filename], 
                                                 shuffle=True, 
                                                 num_epochs=3)
        
        return dataset