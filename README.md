
# Human Activity Recognition with Diffusion Models

This repository contains the code for our Human Activity Recognition (HAR) project, which leverages advanced neural network architectures to analyze and classify types of human activities based on sensor data.


## Data

The dataset required for this project is stored on Google Drive. Please download the dataset from the following link:

[Download Dataset](https://drive.google.com/drive/folders/1swkdEPGvxVEiahi_AYLVbnIWgEAHYSMF?usp=sharing)

### Setting Up Data

After downloading, create a folder named `data` in the main project directory. Place the `human_activity` folder, which you downloaded, inside the `data` folder. This step is crucial for the scripts to locate and use the dataset correctly.


## Running the Code

To run the main project script, execute the following command:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false python main.py --data human_activity
```

This command configures the `XLA_PYTHON_CLIENT_PREALLOCATE` environment variable to `false`, which can be beneficial for managing memory more efficiently when using hardware accelerators like TPUs.


##  Hyperparameter Tuning

You can fine-tune model performance by modifying the training parameters defined in `main.py`. These are implemented using the `absl.flags` library, allowing you to pass arguments via the command line.


