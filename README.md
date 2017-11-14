# RLIE_A3C
Code for the paper 'Speeding up Reinforcement Learning-based Information Extraction Training using Asynchronous Methods' to be presented at EMNLP 2017.

You will need to install [TensorFlow](https://www.tensorflow.org/).

### Data Preparation:

The dev dataset has been used for training the hyperparameters. The test dataset has been used for testing.
* Change to the code directory: `cd code`

#### Create the vectorizers using a pre-trained model:
`python vec_consolidate.py ../data/dloads/Shooter/train.extra 5 trained_model2.p consolidated/vec_train.5.p`<br>
`python vec_consolidate.py ../data/dloads/Shooter/test.extra 5 trained_model2.p consolidated/vec_test.5.p`

#### Consolidate the articles:
`python consolidate.py ../data/dloads/Shooter/train.extra 5 trained_model2.p consolidated/train+context.5.p consolidated/vec_train.5.p`<br>
`python consolidate.py ../data/dloads/Shooter/test.extra 5 trained_model2.p consolidated/test+context.5.p consolidated/vec_test.5.p`


### Running the code:

#### Run the server:
`cd code`<br>
`mkdir consolidated`<br>
`mkdir outputs`<br>
`python server_multiprocessing.py --trainEntities consolidated/train+context.5.p --testEntities consolidated/test+context.5.p --outFile outputs/run.out --modelFile trained_model2.p --entity 4 --aggregate always --shooterLenientEval True --delayedReward False --contextType 2`

#### Run the agent in a separate terminal/tab:
`cd code/a3c`<br>
`mkdir saved_network`<br>
`python a3c.py`


## Cite
```
@inproceedings{sharma2017speeding,
  title={Speeding up Reinforcement Learning-based Information Extraction Training using Asynchronous Methods},
  author={Sharma, Aditya and Parekh, Zarana and Talukdar, Partha},
  booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  pages={2648--2653},
  year={2017}
}
```



## Acknowledgements

[Karthik's DeepRL-InformationExtraction Codebase](https://github.com/karthikncode/DeepRL-InformationExtraction)
[Kosuke Miyoshi's A3C implementation](https://github.com/miyosuda/async_deep_reinforce)
