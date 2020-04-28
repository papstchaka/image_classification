
class preprocessing():
    
    def __init__(self, unique_labels, IMG_SIZE=224, BATCH_SIZE=32):
        self.unique_labels = unique_labels
        self.IMG_SIZE = IMG_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.global_imports()
        
    def global_imports(self):
        global tf
        import tensorflow
        tf = tensorflow
        
    def process_image(self, image_path):
        """
        Takes an image file path and turns the image into a Tensor.
        """
        # Read the image file
        image = tf.io.read_file(image_path)
        # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
        image = tf.image.decode_jpeg(image, channels=3)
        # Convert the colour channel values from 0-255 to 0-1 values
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize the image to deisred value (224, 224)
        image = tf.image.resize(image, size=[self.IMG_SIZE, self.IMG_SIZE])

        return image
    
    def get_image_label(self, image_path, label):
        """
        Takes an image file path name and the assosciated label, processes the image and returns a tuple of (image, label).
        """
        image = self.process_image(image_path)
        return image, label

    def create_data_batches(self, X, y=None, valid_data=False, test_data=False, describe=False, plot=False):
        """
        Creates batches of data out of image (X) and label (y) pairs.
        It shuffles the data if it's training data but doesn't shuffle if it's validation data.
        Also accepts test data as input (no labels).
        """
        # If the data is a test dataset, there are no labels
        if test_data:
            print("Creating test data batches...")
            data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) ##only filepaths (no labels)
            data_batch = data.map(self.process_image).batch(self.BATCH_SIZE)
        # If the data is a valid dataset, no shuffle is needed
        elif valid_data:
            print("Creating validation data batches...")
            data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))) ##only filepaths (no labels)
            data_batch = data.map(self.get_image_label).batch(self.BATCH_SIZE)
        # If the data is a train dataset, shuffle is needed
        else:
            print("Creating training data batches...")
            # Turn filepaths and labels into Tensors
            data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))) ##only filepaths (no labels)
            # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
            data = data.shuffle(buffer_size=len(X))
            # Create (image, label) tuples (this also turns the image path into a preprocessed image)
            data = data.map(self.get_image_label)
            # Turn the training data into batches
            data_batch = data.batch(self.BATCH_SIZE)
        
        if describe:
            print(data_batch.element_spec)
        if plot:
            self.show_25_images(data_batch)
        return data_batch
    
    def show_25_images(self, data):
        import matplotlib.pyplot as plt
        """
        Displays a plot if 25 images and their labels from a data batch
        """
        images, labels = next(data.as_numpy_iterator())
        # Setup the figure
        plt.figure(figsize=(10,10))
        # Loop through 25 (for displaying 25 images)
        for i in range(25):
            # Create subplots (5 rows, 5 columns)
            ax = plt.subplot(5, 5, i+1)
            # Display an image
            plt.imshow(images[i])
            # Add the image label as the title
            plt.title(self.unique_labels[labels[i].argmax()])
            # Turn the grid lines off
            plt.axis("off")
        plt.show()