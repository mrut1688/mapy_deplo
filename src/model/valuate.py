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
        
