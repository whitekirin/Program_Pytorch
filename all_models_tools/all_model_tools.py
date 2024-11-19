from Load_process.file_processing import Process_File
import datetime
import torch

# def attention_block(input):
#     channel = input.shape[-1]

#     GAP = GlobalAveragePooling2D()(input)

#     block = Dense(units = channel // 16, activation = "relu")(GAP)
#     block = Dense(units = channel, activation = "sigmoid")(block)
#     block = Reshape((1, 1, channel))(block)

#     block = Multiply()([input, block])

#     return block

class EarlyStopping:
    def __init__(self, patience=74, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, save_path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, save_path)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        torch.save(model.state_dict(), save_path)
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model to {save_path}")


def call_back(model_name, index, optimizer): 
    File = Process_File()

    model_dir = '../Result/save_the_best_model/' + model_name
    File.JudgeRoot_MakeDir(model_dir)
    modelfiles = File.Make_Save_Root('best_model( ' + str(datetime.date.today()) + " )-" +  str(index) + ".weights.h5", model_dir)
    
    # model_mckp = ModelCheckpoint(modelfiles, monitor='val_loss', save_best_only=True, save_weights_only = True, mode='auto')

    earlystop = EarlyStopping(patience=74, verbose=True) # 提早停止

    reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        factor = 0.94,           # 學習率降低的量。 new_lr = lr * factor
                        patience = 2,                # 沒有改進的時期數，之後學習率將降低
                        verbose = 0,
                        mode = 'min',
                        min_lr = 0                   # 學習率下限
                    )

    return modelfiles, earlystop, reduce_lr