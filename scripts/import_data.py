
class import_data():
    
    def __init__(self, path, NUM_IMAGES = -1):
        self.path = path + "/"
        self.NUM_IMAGES = NUM_IMAGES
        self.global_imports()
        
    def global_imports(self):
        global pd, np, os
        import pandas, numpy, os
        pd, np, os = pandas, numpy, os
        
    def import_raw_data(self, describe = False, plot = False):
        if "labels.csv" in os.listdir(self.path):
            self.labels_data = pd.read_csv(self.path + "labels.csv")

            if describe:
                self.describe()
            if plot:
                self.plot()
            self.get_raw_traindata()
        else:
            self.format_labels()
            self.import_raw_data(describe, plot)
            
    def format_labels(self):
        labels = [folder for folder in os.listdir(self.path + "train/") if os.path.isdir(folder)]
        if len(labels) == 0:
            raise ValueError('parser failed because no `train`-folder found')
        else:
            labelids = []
            labeldata = []
            for label in labels:
                filenames = os.listdir(self.path + "train/" + label + "/")
                labelids.extend([file[:-4] for file in filenames])
                labeldata.extend([label for _ in range(len(filenames))])
            labels_data = pd.DataFrame.from_records(list(zip(labelids, labeldata)), columns=["id","label"])
            labels_data.to_csv(self.path + "labels.csv", index=False)
            
    def describe(self):
        print(self.labels_data.describe())
        print(f'Median amount per unique value in {self.labels_data.columns[-1]} is: {self.labels_data[self.labels_data.columns[-1]].value_counts().median()}')
        self.labels_data[self.labels_data.columns[-1]].value_counts().plot.bar(figsize=(20,10))
        
    def plot(self):
        from IPython.display import display
        from PIL import Image
        display(Image.open(self.path + "train/" + os.listdir(self.path + "train/")[0]))
        
    def get_raw_traindata(self):
        from sklearn.model_selection import train_test_split
        labels = self.labels_data[self.labels_data.columns[-1]]
        self.unique_labels = np.unique(labels)
        X = [f'{self.path}train/{fname}.jpg' for fname in self.labels_data[self.labels_data.columns[0]]]
        y = [label == self.unique_labels for label in labels]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X[:self.NUM_IMAGES], y[:self.NUM_IMAGES], test_size=0.2, random_state=42)
        