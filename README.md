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
> Pipeline supports two types of folder structures to work with:
> 1. already 'manually' prepared
> /<dataset-name>
> ---/train/
> ---/test/
> ---/labels.csv

> where as `train` contains all training pictures (directly in train-folder, no subfolders), `test` contains all test pictures (directly in test-folder, no subfolders) and `labels.csv` contains the information of train-pictures labels. Contains `id`-column (that contains the name of the training picture file) and `label` (corresponding label of the file)

> 2. 'Tensorflow'-alike folder structure
> /<dataset-name>
> ---/train
> ---/---/<label1>
> ---/---/<label2>
> ---/---/...
> ---/---/<labelx>
> ---/validation
> ---/---/<label1>
> ---/---/<label2>
> ---/---/...
> ---/---/<labelx>
> ---/test

> where as only one of `test`- or `validation`-folder must be available --> Pipeline is able to handle all three folder structures (given `test` AND `validation`, given `test` and NO `validation` and given NO `test` but `validation`).
> Pipeline will than change the folder structure to be exact the same as in 1.

> IMPORTANT: Folder structure doesn't need to provide a validation set. This set will automatically split the training data into training and validation set using sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html" target="_blank">`train_test_split`</a>-function.