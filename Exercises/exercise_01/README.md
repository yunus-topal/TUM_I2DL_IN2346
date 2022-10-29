# Introduction to Deep Learning (IN2346)

# Technical University Munich - WS 2022/23

## Anaconda setup

The enviroment that we are going to use throughout this course is Anaconda. 

Download and install it conda, from https://www.anaconda.com/. 

Open a terminal and create an environment using the command:

`conda create --name i2dl python=3.10 -y`

Next, activate the environment using the command:

`conda activate i2dl`

Within the terminal, direct yourself to exercise_01's folder and continue with installation of requirements and starting jupyter notebook as mentioned above, i.e.

`pip install -r requirements.txt` 

`jupyter notebook`


## Exercise Download

The exercises will be uploaded to the course website at https://www.3dunderstanding.org/i2dl-w22/. You can download the exercises directly from there.

Each exercise contains at least one jupyter notebook, that could be opened by the jupyter-notebook plaform (In the terminal, go to the relevant folder and type `jupyter notebook`), or severla IDEs that support it,
such as Microsoft's VScode or JetBrains' PyCharm.

The rest of the code resides in .py files. Access those files in any IDE of your choice (VScode, Pycharm, Spyder). You could also work directly on the jupyt plaform, but we do not recommend it.
IDE is a powerful tool that allows you to navigate easily thorugh the projects, debug and even shows you your errors.

## Google Colab

Deep learning is an expensive practice. It only bursted about 10 years ago into our lives because GPUs became strong enough to allow the magic it is.
As most of us do not posses a computer that has a GPU, google offers a free platform, that allows you to use their cloud GPUs. Weak as they might be, they are still powerful enough
to ease our training processes and make them 10x faster. This will be crucial towards the later exercises of the course. Therefore, we recommend you to become fimiliar with it early.
However, exercises 1-5 do not require such capabilites.

In order to use the platform, open a folder in your goolge-drive main page, under the name `i2dl`, for consistency with the exercises.
In there, paste the exercises folders. Then, you could simply open the notebooks with the colab-notebook. There, you should follow the instructions we've assembled for you in each notebook.

Pay attention that files require a few seconds, in order to save to the colab cloud disk. Therefore, run the zipping cell in the notebook after you've waited a few seconds, letting the previous cell
save your models to the disk. Otherwise, you will encounter some troubles, trying to submit your code without your models.

Download your zipped exercise from the drive and submit it to the submission platform.

NOTE: Pytorch does NOT support MacBooks with the M1 or M2 cpus. Therefore, in order to utilize a GPU --> use colab.

### The directory layout for the exercises

    i2dl_exercises
    ├── datasets                   # The datasets required for all exercises will be placed here
    ├── exercise_1                 
    ├── exercise_2                     
    ├── exercise_3                    
    ├── exercise_4
    ├── exercise_5
    ├── exercise_6
    ├── exercise_7                              
    ├── exercise_8
    ├── exercise_9
    ├── exercise_10
    ├── exercise_11
    ├── exercise_12                    
    ├── LICENSE
    └── README.md


## 4. Dataset Download

Datasets will generally be downloaded automatically by exercise notebooks and stored in a common datasets directory shared among all exercises. A sample directory structure for cifar10 dataset is shown below:-

    i2dl_exercises
        ├── datasets                   # The datasets required for all exercises will be downloaded here
            ├── cifar10                # Dataset directory
                ├── cifar10.p          # dataset files 

## 5. Exercise Submission
Your trained models will be automatically evaluated on a test set on our server. To this end, login or register for an account at:

https://i2dl.vc.in.tum.de/

Note that only students who have registered for this class in TUM Online can register for an account. This account provides you with temporary credentials to login onto the machines at our chair.

At the end of each exercise there is a script that zips all of the relevant files. All your trained models should be inside `models` directory in the exercise folder. Make sure they are there, especially while working with google-colab.

Then, submit it to the submission server (should not include any datasets). 

You can login to the above website and upload your zip submission. Your submission will be evaluated by our system. 

You will receive an email notification with the results upon completion of the evaluation. To make the exercises more fun, you will be able to see a leaderboard of everyone's (anonymous) scores on the login part of the submission website.

Note that you can re-evaluate your models until the deadline of the current exercise. Whereas the email contains the result of the current evaluation, the entry in the leader board always represents the best score for the respective exercise.


## 6. Acknowledgments

We want to thank the **Stanford Vision Lab**, **PyTorch** and **PyTorch Lightning** for allowing us to build these exercises on material they had previously developed. We also thank the **TU Munich Dynamic Vision and Learning Group** for helping create course content.
