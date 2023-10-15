# Final Project

This is a **group assignment**.

## Code Implementation & Technical Report

The final deliverables include a 4-page IEEE-format report, code implementation and a detailed GitHub readme file.

The final project is due Tuesday, December 6 @ 11:59 PM. Find the complete [rubric](https://ufl.instructure.com/courses/455013/assignments/5244219) in the Canvas assignment.

## Training Data

The training data set is the same for every team in this course.

You can download the training data from the Canvas page:

* ["data_train.npy"](https://ufl.instructure.com/files/72247539/download?download_frd=1)
* ["t_train.npy"](https://ufl.instructure.com/files/72245951/download?download_frd=1)

## Deliverables

The Technical Report for the project can be found at `EEL_5840_Final_Project_POW_9001.pdf`.

The code implementation is under `model_4`. The instruction below all deal with files under this directory.

The setup steps for training are as follows:
1. Set up file paths for the location of the training data and the corrected labels in the `data_work` function in `model_4_train.py`.
1. Run the training file `model_4_train.py` through the command line as `python3 model_4_train.py`.
1. This file will generate the following a file contianing the mean as `data_mean.npy` and a file containing the standard deviation as `data_std.npy`.
1. The model weights will also be saved, along with the accuracy and loss function graphs.

The setup steps for testing are as follows:
1. Add the file path of the weights of the trained model, the standard deviation file, the mean file, the testing dataset, and the labels of the testing dataset to the `run` function under  `model_4_test.py`.
1. Run the testing file `model_4_test.py` through the command line as `python3 model_4_test.py`.
1. This will produce the accuracy and confusion matrix of the model.

NOTE: The weights of the trained model can be found under the file `model_4_wts_link`.
