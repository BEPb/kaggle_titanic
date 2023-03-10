<div align="center">


<img src="./art/logo.png" alt="Bot logo" width="300" height="156.5">

# kaggle titanic

</div>

## How it works?

It's very simple: here are the solutions for the [titanic competition ](https://www.kaggle.com/competitions/titanic)

## Order of preparation and work

1. Clone the repository or download the archive from github or using the following commands on the command line
    ```command line
    $cmd
    $ git clone https://github.com/BEPb/kaggle_titanic
    $ cd kaggle_titanic
    ```

2. Create a Python virtual environment.
3. Install all necessary packages for our code to work using the following command:

     ```
     pip install -r requirements.txt
     ```
4. file list
- data - directory with data files
- data/titanic.zip - archive of the initial tabular data of the competition (3 files)
- data/gender_submission.csv - one of the original data files
- data/test.csv - one of the original data files
- data/train.csv - one of the original data files
- notebooks - directory with jupiter notebooks
- notebooks/eda_and_analysis - directory with notebooks food and data analysis
- notebooks/eda_and_analysis/titanic_universal_eda.ipynb - universal food notebook
- notebooks/eda_and_analysis/titanic_eda.ipynb - food notebook
- notebooks/solutions - directory with notebook solutions
- python_code - directory with python code solutions


5. Well, as a result of the training, I wrote a console application that, based on the model, predicts whether the 
passenger whose data you enter in the fields will survive or not. 
   - python_code/Titanic_gui.py

prediction for my data, let's say I'm traveling first class with my family:
<img src="./art/gui.png" alt="Gui logo" width="600" height="600">


prediction for data from a set:

<img src="./art/gui2.png" alt="Gui logo" width="600" height="600">