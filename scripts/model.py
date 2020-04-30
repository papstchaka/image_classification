# start tensorboard with `tensorboard --logdir='{path}/logs' --port 2222`


class model():
    
    def __init__(self, unique_labels, path, IMG_SIZE=224, MODEL_URL="https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"):
        self.MODEL_URL = MODEL_URL
        self.INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # batch, height, width, colour channels
        self.OUTPUT_SHAPE = len(unique_labels)
        self.path = path + "/"
        self.global_imports()
        print("TF version:", tf.__version__)
        print("TF Hub version:", hub.__version__)

        # Check for GPU availability
        print("GPU", "available" if tf.config.list_physical_devices("GPU") else "not available")
        
    def global_imports(self):
        global tf, hub, datetime, os
        import tensorflow, tensorflow_hub, datetime, os
        tf, hub, datetime, os = tensorflow, tensorflow_hub, datetime, os
        
    def create_model(self, describe=False):
        print("Building model with:", self.MODEL_URL)

        # Setup the model layers
        model = tf.keras.Sequential([
                    hub.KerasLayer(self.MODEL_URL), # Layer 1 (input layer)
                    tf.keras.layers.Dense(units = self.OUTPUT_SHAPE,
                                          activation = "softmax") # Layer 2 (output layer)
        ])

        # Compile the model
        model.compile(
              loss = tf.keras.losses.CategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ["accuracy"]
          )

        # Build the model
        model.build(self.INPUT_SHAPE)
        
        if describe:
            print(model.summary())
        return model
    
    def create_tensorboard_callback(self):
        # Create a log directory for storing TensorBoard logs
        logdir = os.path.join(self.path + "logs/", 
                             # Make it so the logs get tracked whenever an experiment gets run
                             datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                             )
        logdir = logdir.replace("/","\\")
        return tf.keras.callbacks.TensorBoard(logdir, update_freq="batch")
    
    def early_stopping_callback(self):
        return tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                patience=3)
    
    def train_model(self, train_data, val_data, NUM_EPOCHS=100, model_name=""):
        """
        Trains a given model and returns the trained version.
        """
        # Create a model
        model = self.create_model()

        # Create a new TensorBoard session everytime a model gets trained
        tensorboard = self.create_tensorboard_callback()
        early_stopping = self.early_stopping_callback()

        # Fit the model to the data passing it the created callbacks
        model.fit(x = train_data,
                  epochs = NUM_EPOCHS,
                  validation_data = val_data,
                  callbacks = [tensorboard, early_stopping]
                  )
        # Return the fitted model
        model_name = model_name if len(model_name) > 0 else datetime.datetime.now().strftime("%H%M%S")
        self.save_model(model, model_name)
        model.evaluate(val_data)
        return model
    
    def save_model(self, model, suffix=None):
        """
        Saves a given model in a models directory and appends a suffix (string)
        """
        # Create a model diretory pathname with current time
        modeldir = os.path.join(self.path + "models/",
                                   datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model_path = modeldir + "-" + suffix + ".h5" # save format of model
        print(f'Saving model to: {self.model_path} ...')
        model.save(self.model_path)
        return self.model_path

    def load_model(self, model_path=""):
        """
        Loads a saved model from a specified path
        """
        model_path = model_path if len(model_path) > 0 else self.model_path
        print(f'Loading saved model from: {model_path}')
        model = tf.keras.models.load_model(model_path,
                                        custom_objects = {"KerasLayer" : hub.KerasLayer})
        return model