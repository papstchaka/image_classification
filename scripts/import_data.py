
class import_data():
    
    def __init__(self, path, NUM_IMAGES=-1):
        self.path = path + "/"
        self.NUM_IMAGES = NUM_IMAGES
        self.global_imports()
        
    def global_imports(self):
        global pd, np, os
        import pandas, numpy, os
        pd, np, os = pandas, numpy, os
        
    def check_helper(self, foldername, folders):
        for folder in folders:
            for file in os.listdir(self.path + foldername + "/" + folder):
                os.rename(f'{self.path}{foldername}/{folder}/{file}',f'{self.path}{foldername}/{file}')                
            os.rmdir(self.path + foldername + "/" + folder)
        
    def check_reformat_folder_structure(self):
        # Check test folder / change validation folder to test folder
        test, validation = False, False
        if "test" in os.listdir(self.path):
            test = True
        if "validation" in os.listdir(self.path):
            validation = True
            val_folders = [folder for folder in os.listdir(self.path + "validation/") if os.path.isdir(self.path+"validation/"+folder)]
            val_folders = val_folders if len(val_folders) > 0 else [""]
        
        if validation and (test == False):
            self.check_helper("validation", val_folders)
            os.rename(self.path + "validation", self.path + "test")
        
        if (validation == False) and (test == False):
            raise ValueError('no data for testing purposes found')
            
        # Check train folder
        folders = [folder for folder in os.listdir(self.path + "train/") if os.path.isdir(self.path+"train/"+folder)]
        if len(folders) > 0:
            self.check_helper("train", folders)
            
    def import_raw_data(self, describe=False, plot=False):
        if "labels.csv" in os.listdir(self.path):
            self.labels_data = pd.read_csv(self.path + "labels.csv")

            if describe:
                self.describe()
            if plot:
                self.plot()
            self.check_reformat_folder_structure()
            self.get_raw_traindata()
        else:
            self.format_labels()
            self.import_raw_data(describe, plot)
            
    def format_labels(self):
        labels = [folder for folder in os.listdir(self.path + "train/") if os.path.isdir(self.path+"train/"+folder)]
        if len(labels) == 0:
            raise ValueError('parser failed because no `train`-folder found')
        else:
            labelids = []
            labeldata = []
            for label in labels:
                filenames = os.listdir(self.path + "train/" + label + "/")
                labelids.extend([file for file in filenames])
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
        X = [f'{self.path}train/{fname}' for fname in self.labels_data[self.labels_data.columns[0]]]
        y = [label == self.unique_labels for label in labels]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X[:self.NUM_IMAGES], y[:self.NUM_IMAGES], test_size=0.2, random_state=42)
        