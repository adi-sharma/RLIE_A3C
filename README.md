# RLIE_A3C
Code for the paper 'Speeding up Reinforcement Learning-based Information Extraction Training using Asynchronous Methods' to be presented at EMNLP 2017.

## Data Preparation:

The dev dataset has been used for training the hyperparameters. The test dataset has been used for testing.

### Create the vectorizers using a pre-trained model:
python vec_consolidate.py ../data/dloads/Shooter/train.extra 5 trained_model2.p consolidated/vec_train.5.p
python vec_consolidate.py ../data/dloads/Shooter/test.extra 5 trained_model2.p consolidated/vec_test.5.p

### Consolidate the articles:
python consolidate.py ../data/dloads/Shooter/train.extra 5 trained_model2.p consolidated/train+context.5.p consolidated/vec_train.5.p
python consolidate.py ../data/dloads/Shooter/test.extra 5 trained_model2.p consolidated/test+context.5.p consolidated/vec_test.5.p


## Running the code:

### Run the server:
* cd code
* mkdir consolidated
* mkdir outputs
* python server_multiprocessing.py --trainEntities consolidated/train+context.5.p --testEntities consolidated/test+context.5.p --outFile outputs/run.out --modelFile trained_model2.p --entity 4 --aggregate always --shooterLenientEval True --delayedReward False --contextType 2

### Run the agent in a separate terminal/tab:
* cd a3c
* mkdir saved_network
* python a3c.py
