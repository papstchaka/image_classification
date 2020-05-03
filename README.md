# image_classification
This repository provides an end-to-end pipeline for tensorflow-based image classification using neural network.

<h2 align="center">
  <img src=https://github.com/papstchaka/image_classification/blob/master/assets/prediction_example.jpg alt="Prediction Example" width="800px" />
</h2>

## Requirements
* pandas
* numpy
* pillow
* sklearn
* tensorflow
* matplotlib
* tensorflow_hub
* Jupyter Notebook with IPython - for displaying purposes

> install via `pip install pandas numpy pillow sklearn tensorflow matplotlib tensorflow_hub jupyter notebook`

## Fork project and set it up to work on local laptop
* Fork/Clone the repository to your local machine into a folder like `image_classification`, go to that folder and open `classify_images.ipynb`. This Notebook provides an example of the whole pipeline.
* To use the pipeline, you'll definitely have to change the first line `path = "D:/datasets/horse-or-human"` to the corresponding dataset you have on your own machine like `path = "path-to-dataset"`.
* Pipeline supports two types of folder structures to work with:
```
    1. already 'manually' prepared

        /<dataset-name>
        ---/train/
        ---/test/
        ---/labels.csv
```

where as `train` contains all training pictures (directly in train-folder, no subfolders), `test` contains all test pictures (directly in test-folder, no subfolders) and `labels.csv` contains the information of train-pictures labels. Contains `id`-column (that contains the name of the training picture file) and `label` (corresponding label of the file)

```
    2. 'Tensorflow'-alike folder structure

        /<dataset-name>
        ---/train
        ---/---/<label1>
        ---/---/<label2>
        ---/---/...
        ---/---/<labelx>
        ---/validation
        ---/---/<label1>
        ---/---/<label2>
        ---/---/...
        ---/---/<labelx>
        ---/test
```

where as only one of `test`- or `validation`-folder must be available --> Pipeline is able to handle all three folder structures (given `test` AND `validation`, given `test` and NO `validation` and given NO `test` but `validation`).
Pipeline will than change the folder structure to be exact the same as in 1.

* IMPORTANT: Folder structure doesn't need to provide a validation set. This set will automatically split the training data into training and validation set using sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html" target="_blank">`train_test_split`</a>-function.

## How to use it / possible modifications
Pipeline provides various options to use it. It is built very modular, so you can insert and use all the scripts in `script`-folder independently. Going through them step-by-step

### import_data.py
Manages the import of training data and labels, splitting it into training and validation sets.
#### init class
inits the class
* required parameter
```
    path:           - path to the dataset (as described above) - String
```
* optional parameter
```
    NUM_IMAGES:     - number of images that shall be used to train the model. Helps during testing process to avoid the model from taking too much time during training - Integer - default = -1 --> all
```
#### import_raw_data()
refactors the required folder structure and imports the data from the files
* required parameter
```
    none
```
* optional parameter
```
    describe:       - whether or not the Pipeline shall give a short description of the given labels and show the distribution of given labels - Boolean - default = False
    plot:           - whether or not showing an example of the training data. Will show the first image from the train-folder - Boolean - default = False
```
* returns
```
    none
```
#### get_raw_traindata()
Loads the filenames of the training data and splits it into X_train, X_val, y_train and y_test. Loads labels and gives a list of all unique labels that exist.
* required parameter
```
    none
```
* optional parameter
```
    none
```
* returns
```
    none
```

### preprocessing.py
Preprocesses the training and validation (or testing) datasets, gives them back as data batches to be used during training
#### init class
inits the class
* required parameter
```
   unique_labels:   - list of all unique labels that exist. Should be the list that is given by `import_data.unique_labels` which is created using `import_data.get_raw_traindata()` - numpy.array
```
* optional parameter
```
    IMG_SIZE:       - size that the images must have to be processed by the model (is defined by the size that the desired model from `tensorflow_hub` uses) - Integer - default = 224
    BATCH_SIZE:     - size of data batches that shall be processed during training - Integer - default = 32
```
#### create_data_batches()
Creates and returns data batches from given data. Optionally describes and displays an example
* required parameter
```
    X:              - X data for batch (like X_train, X_val or X_test) - pandas.DataFrame
```
* optional parameter
```
    y:              - y data for batch (like y_train, y_val). Has to be provided if training or validation batches shall be returned. Only unused for y_test because no y_test available - pandas.DataFrame - default = None
    valid_data:     - whether or not validation batch shall be returned - Boolean - default = False
    test_data:      - whether or not test batch shall be returned - Boolean - default = False
    describe:       - whether or not the Pipeline shall give a short description of the determined data batch - Boolean - default = False
    plot:           - whether or not showing an example of the determined data batch. Will show the first 25 images from the batch in one plot - Boolean - default = False
```
* returns
```
    data_batch:     - data batch of provided data - tensorflow.data.Dataset.batch
```

### model.py
Creates, trains, loads and saves the classification model
#### init class
inits the class
* required parameter
```
   unique_labels:   - list of all unique labels that exist. Should be the list that is given by `import_data.unique_labels` which is created using `import_data.get_raw_traindata()` - numpy.array
   path:            - path where logs and model shall be saved (can be same as <path-to-dataset>) - String
```
* optional parameter
```
    IMG_SIZE:       - size that the images must have to be processed by the model (is defined by the size that the desired model from `tensorflow_hub` uses) - Integer - default = 224
    MODEL_URL:      - url to the desired pretrained model from <a href="https://www.tensorflow.org/hub" target="_blank">`TensorFlow-Hub`</a> - String - default = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
```
#### train_model()
creates, trains, evaluates and saves the model. An own created model can be passed if desired.
* required parameter
```
    train_data:     - data to train model on (given by preprocessing.py) - tensorflow.data.Dataset.batch
    val_data:       - data for model validation (given by preprocessing.py) - tensorflow.data.Dataset.batch
```
* optional parameter
```
    NUM_EPOCHS:     - Maximum number of training epochs. Model will stop (thanks to early stopping callback) to train if no new progress is made, to avoid overfitting - Integer - default = 100
    model_name:     - name of file where model will be stored (path of model is constructed by - in init - given path and a new folder of current date + given model_name as suffix) - String - default = "" (will result in filename of %H%M%S.h5)
    model:          - own, custom made model can be used instead of new one - tensorflow.keras.Sequential - default = None (which means that Pipeline will create new)
    describe:       - whether or not to describe the how the model looks like - Boolean - default = False
```
* returns
```
    model:          - the trained and evaluated model - tensorflow.keras.Sequential
```

### predict.py
Predicts and shows validation data. Can only handle data which has labels (like validation or training data batches)
#### init class
inits the class
* required parameter
```
    model:          - model that shall predict the data - tensorflow.keras.Sequential
    data:           - data batches to be predicted (most likely the validation dataset, given by preprocessing.py) - tensorflow.data.Dataset.batch 
    unique_labels:  - list of all unique labels that exist. Should be the list that is given by `import_data.unique_labels` which is created using `import_data.
```
* optional parameter
```
    none
```
#### check_predictions()
Predicts labels based on data and evaluates whether right or wrong. Plots result
* required parameter
```
    none
```
* optional parameter
```
    num_rows:       - number of rows that will be plotted (multiplied with num_cols results in the number of predictions that are plotted) - Integer - default = 3
    num_cols:       - number of columns that will be plotted (multiplied with num_rows results in the number of predictions that are plotted) - Integer - default = 2
    i_multiplier:   - number of images that are skipped before starting to plot (0 means, the first x images in the data batch are used, 10 means that the images 10 to 10+x will be used) - Integer - default = 0
```
* returns
```
    none
```

## What's next?

## Further informations:
