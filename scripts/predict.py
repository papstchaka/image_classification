
class predict():
    
    def __init__(self, model, data, unique_labels):
        self.model = model
        self.unique_labels = unique_labels
        self.global_imports()
        self.predict(data)
        
    def global_imports(self):
        global np, plt
        import numpy, matplotlib.pyplot
        np, plt = numpy, matplotlib.pyplot
    
    def predict(self, data):
        self.predictions = self.model.predict(data, verbose=1)
        self.val_images, self.val_labels = self.unbatchify(data)
        
    def get_pred_label(self, prediction_probabilities):
        """
        Turns an array of prediction probabilities into a label
        """
        return self.unique_labels[np.argmax(prediction_probabilities)]
    
    def unbatchify(self, data):
        """
        Takes a batched dataset of (image, label) Tensors and returns seperate arrays of images and labels
        """
        images = []
        labels = []
        # Loop through unbatched data
        for image, label in data.unbatch().as_numpy_iterator():
            images.append(image)
            labels.append(self.unique_labels[np.argmax(label)])
        return images, labels
    
    def plot_pred(self, prediction_probabilities, labels, images, n=1):
        """
        View the prediction, ground truth and image for sample n
        """
        pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

        # Get the pred label
        pred_label = self.get_pred_label(pred_prob)

        # Plot image & remove ticks
        plt.imshow(image)
        plt.xticks()
        plt.yticks()

        # Change the colour of the title depending on if the prediction is right or wrong
        if pred_label == true_label:
            color = "green"
        else:
            color = "red"

        # Change plot title to be predicted, probability of prediction and truth label
        plt.title("{} with probability of {:2.0f}%. It is actually a {}".format(pred_label, np.max(pred_prob)*100, true_label), color=color)
    
    def plot_pred_conf(self, prediction_probabilities, labels, n=1):
        """
        Plot the top 10 highest prediction confidences along with the truth label for sample n
        """
        pred_prob, true_label = prediction_probabilities[n], labels[n]

        # Get the predicted label
        pred_label = self.get_pred_label(pred_prob)

        # Find the top 10 prediction confidence indexes 
        top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
        # Find the top 10 prediction confidence values
        top_10_pred_values = pred_prob[top_10_pred_indexes]
        # Find the top 10 prediction labels
        top_10_pred_labels = self.unique_labels[top_10_pred_indexes]

        # Setup plot
        top_plot = plt.bar(np.arange(len(top_10_pred_labels)),
                         top_10_pred_values,
                         color="grey")
        plt.xticks(np.arange(len(top_10_pred_labels)),
                 labels=top_10_pred_labels,
                 rotation="vertical")

        # Change color of true label
        if np.isin(true_label, top_10_pred_labels):
            top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
    
    def check_predictions(self, num_rows=3, num_cols=2, i_multiplier=0):
        # Check a few predictions and their different values
        num_images = num_rows * num_cols
        plt.figure(figsize=(10*num_cols, 5*num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            self.plot_pred(prediction_probabilities = self.predictions,
                    labels = self.val_labels,
                    images = self.val_images,
                    n = i + i_multiplier)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            self.plot_pred_conf(prediction_probabilities = self.predictions,
                         labels = self.val_labels,
                         n = i + i_multiplier)
        plt.tight_layout(h_pad=1.0)
        plt.show()