"""
This is the Ymap neural algorithm design using Tensorflow. 
It will be designed using depthwise convolutional neural networks. 
This model was designed to to inputted on a raspberry pi by converting it to a JSON.
contributors:All 4 group members.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History


def Ymap(nb_classes, Chans = 4, Samples = 178,dropoutRate = 0.5, kernLength = 89, F1 = 8,D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """     Inputs:      nb_classes      : int, number of classes to classify.
      Initially it was 6 but yash says  
      remove the random class as it can interfere in classfication  
      Chans, Samples  : number of channels and time points in the EEG data     
      dropoutRate     : dropout fraction     
      kernLength      : length of temporal convolution in first layer. I have set
      this half the sampling rate worked well in practice.
      Known from research paper.    
      F1, F2          : number of temporal filters (F1) and number of pointwise
      filters (F2) to learn.       
      D               : number of spatial filters to learn within each temporal                        
      convolutiondropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """        

 
    if dropoutType == 'SpatialDropout2D':        
        dropoutType = SpatialDropout2D    
    elif dropoutType == 'Dropout':        
        dropoutType = Dropout    
    else:        
        raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')
    input1   = Input(shape = (Chans, Samples, 4))

    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                          input_shape = (Chans, Samples, 4),
                          use_bias = False)(input1)    
    block1       = BatchNormalization()(block1)    
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False,
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    block2       = SeparableConv2D(F2, (1, 16),                                   
                                   use_bias = False, padding = 'same')(block1)    
    block2       = BatchNormalization()(block2)    
    block2       = Activation('elu')(block2)    
    block2       = AveragePooling2D((1, 8))(block2)    
    block2       = dropoutType(dropoutRate)(block2)        
    flatten      = Flatten(name = 'flatten')(block2)
    dense        = Dense(nb_classes, name = 'dense',                         
                         kernel_constraint = max_norm(norm_rate))(flatten)    
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    
    return Model(inputs=input1, outputs=softmax)

class PlotMetrics(History):   
    def on_train_begin(self, logs=None):        
        self.epoch = []        
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):        
        logs = logs or {}        
        self.epoch.append(epoch)        
        for k, v in logs.items():            
            self.history.setdefault(k, []).append(v)        
        clear_output(wait=True)        
        plt.figure(figsize=(12, 6))        
        plt.subplot(1, 2, 1)        
        plt.plot(self.epoch, self.history['accuracy'], label='Training Accuracy')        
        plt.plot(self.epoch, self.history['val_accuracy'], label='Validation Accuracy')        
        plt.title('Accuracy')        
        plt.xlabel('Epoch')        
        plt.ylabel('Accuracy')        
        plt.legend()
        plt.subplot(1, 2, 2)        
        plt.plot(self.epoch, self.history['loss'], label='Training Loss')        
        plt.plot(self.epoch, self.history['val_loss'], label='Validation Loss')        
        plt.title('Loss')        
        plt.xlabel('Epoch')        
        plt.ylabel('Loss')        
        plt.legend()        
        plt.tight_layout()        
        plt.show()


mapy=Ymap(5)
mapy.save("mapy")
