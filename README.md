# Text Classification

## Background

We are building a patient outcome prediction model based on tabular data inputs. The tabular inputs indicate the patient records are represented by a row stored in a table, e.g.,

| uid  | age  | gender | label |
| ---- | ---- | ------ | ----- |
| 1    | 20   | female | 0     |
| 2    | 32   | male   | 1     |

each row is a patient and we need to make a prediction for the targeting label taking the input feature `age` and `gender` as the input. In this project , we provide a complete pipeline for a tabular prediction problem including

- Data Preprocessing
- Feature Engineering
- Evaluation

###  Step I: Data Preprocessing

This step is completed by `data_preprocessing.py`, including: 

- manually check the data and assign the columns into three different types: binary, numerical, and categorical; save the results in three lists. The lists are saved in `feature_distribution.py`. 
- fill NaN values with appropriate techniques considering the feature types.

### Step 2: Tabular Captioning

In this step, we transforms each row of table to its corresponding textual descriptions, e.g., for the tabular data

| uid  | age  | gender | smoking | label |
| ---- | ---- | ------ | ------- | ----- |
| 1    | 20   | female | 0       | 1     |
| 2    | 32   | male   | 1       | 0     |

we can obtain two sentences, one for each patient:

- `'age 20; gender female'`
- `'age 32; gender male; smoking'`

It should be noted during the processing you will need to consider the feature types. For `numerical` and `categorical` features, we need to concatenate the column names and the cell values; for `binary` features, we only keep the column when its value is `1`.

### Step 3: model development

In this step, we develop a neural network `TextMLP`  with an embedding layer and fully-connected layers to make text classification. Consider the captioning of the raw patient record, e.g., `'age 20; gender female'`, we know the corresponding label is `1`. This reduces to a text classification problem for prediction the patient outcome. Here we provide two versions: with pretrained or unpretrained embedding layer.



### Step 4: model training and evaluation

In this step, we implement the training function of the model `TextMLP` for binary classification in `text_classification.py`.  `run_textmlp.py` and `run_pretrained_textmlp.py` are main scripts of running training and evalution of the two versions of `TextMLP`. The full training pipeline includes

- #### Train/test split

  We use the function: `sklearn.model_selection.train_test_split` to split the data, 20% for test and 80% for training.

- #### Tokenization of the input sentences: 

  We use the pretrained tokenizers provided by [`transformers`](https://github.com/huggingface/transformers) to tokenize the input sentences to discrete `index input_ids` and `attention_mask`. 

- #### Training of the model

  We provide a simple visualization function: `plot_loss` ( contained in  `text_classification.py`) to analyze the training of models. 

- #### Evaluation of the model's performance on the test set