
class model():
    
    def __init__(self):
        self.global_imports()
        print("TF version:", tf.__version__)
        print("TF Hub version:", hub.__version__)

        # Check for GPU availability
        print("GPU", "available" if tf.config.list_physical_devices("GPU") else "not available")
        
    def global_imports(self):
        global tf, hub
        import tensorflow, tensorflow_hub
        tf, hub = tensorflow, tensorflow_hub