# listnet-listmle-infer

## About

Notebook-based clone of [allegro/allRank](https://github.com/allegro/allRank) used to generate ListNet and ListMLE prediction scores for ranking senior men's and women's 100 m sprint athletes. It loads the trained weights produced in [listmle_listnet_train](https://github.com/mphop-T/listmle_listnet_train) and scores the 2024 test files. No training happens in this repo.

This is **stage 3** of the ListNet/ListMLE pipeline, which is spread over three repos:

| Stage | Step | Repo |
|---|---|---|
| 1 | Data preparation and LibSVM export | [athletes-ranking-general](https://github.com/mphop-T/athletes-ranking-general) |
| 2 | Tuning and training (Docker), producing a `model.pkl` and `used_config.json` per run | [listmle_listnet_train](https://github.com/mphop-T/listmle_listnet_train) |
| 3 | Scoring the men's and women's 2024 test sets with the selected checkpoints | **listnet-listmle-infer** (this repo) |
| 4 | Evaluation (NDCG, Spearman) and significance testing for all four models | [athletes-ranking-general](https://github.com/mphop-T/athletes-ranking-general) |

## What this repo does

For each of the four models (ListMLE men, ListMLE women, ListNet men, ListNet women), a notebook:

1. Reads the `used_config.json` of the run selected during tuning in listmle_listnet_train and points its `data.path` at the gender folder holding the test file. The edited copy is saved in `scripts/`.
2. Rebuilds the network from that configuration with `make_model` and loads the saved weights from `model.pkl` into it with `load_state_dict`. The weights are those of the final training epoch.
3. Loads the 2024 test file (`test.txt`, LibSVM format, one query per event) and pads every event to the length of the longest event.
4. Passes the test data through the network in evaluation mode with gradients switched off, so that every athlete receives one real-valued score.
5. Writes the scores to a text file, one event per line. Positions that are padding receive a score of minus infinity and must be discarded when the scores are matched back to athletes.

The athletes are not ranked here. Ranking within each event, NDCG, Spearman's correlation and the significance tests are done in [athletes-ranking-general](https://github.com/mphop-T/athletes-ranking-general), where the score files are attached to the athlete rows restored from the LibSVM file in the same order.

## Requirements

- Python 3.9.23
- The packages in `requirements.txt` (PyTorch, pandas, allRank dependencies)

## Inputs

| Input | Source |
|---|---|
| `model.pkl` and `used_config.json` of the selected run for each model and gender | [listmle_listnet_train](https://github.com/mphop-T/listmle_listnet_train) |
| `test.txt` per gender folder (`athletes_men_data_combined/`, `athletes_women_data_combined/`) | Data-preparation notebook in [athletes-ranking-general](https://github.com/mphop-T/athletes-ranking-general) |

The selected runs are:

| Model | Gender | Selected run |
|---|---|---|
| ListMLE | Men | `listmle_male_t0` |
| ListMLE | Women | `listmle_women_t0` |
| ListNet | Men | `listnet_men_t28` |
| ListNet | Women | `listnet_women_t28` |

Copy each run's `model.pkl` and `used_config.json` into this repo before running the notebooks. The notebooks load `used_config.json` by path, so keep each run in a folder named after the run.

## Outputs

One score file per model and gender:

| File | Model | Gender |
|---|---|---|
| `allrank_listmle_male_combined_tensor_output.txt` | ListMLE | Men |
| `allrank_listmle_women_combined_tensor_output.txt` | ListMLE | Women |
| `allrank_listnet_male_combined_tensor_output.txt` | ListNet | Men |
| `allrank_listnet_women_combined_tensor_output.txt` | ListNet | Women |

Each line holds the scores of one event, in the order of the events in `test.txt`. These files are copied into athletes-ranking-general for evaluation.

## Usage

1. Create a Python 3.9.23 environment and install the dependencies from `requirements.txt` (PyTorch, pandas and the allRank package). Training does not need Docker, so a normal Python environment is enough. Run the notebooks in a kernel for this environment. [athletes-ranking-general](https://github.com/mphop-T/athletes-ranking-general) uses a different one (Python 3.13.5), so the two repos should not share a kernel.
2. Copy the selected `model.pkl` and `used_config.json` for one model and gender into the repo, and put the matching `test.txt` in the gender data folder.
3. Open the notebook for that model and gender and run all cells. The first code cell that uses `used_config.json` writes the edited config to `scripts/`.
4. Repeat for the other three models and genders, then copy the four score files to athletes-ranking-general.

Note that `model.pkl` is a Python pickle. Only load checkpoints you trained yourself or received from a source you trust.

## Modifications from upstream

This repo is a clone of [allegro/allRank](https://github.com/allegro/allRank) (Pobrotyn et al., 2020, [arXiv:2005.10084](https://arxiv.org/abs/2005.10084)), used under its Apache 2.0 license. The `allrank` package is used as it is. The additions are the inference notebooks and the helper functions inside them, which load a LibSVM role from a local Windows path, pad every event to the longest event in the file, and score the data without computing a loss.

## License

Apache License 2.0, inherited from upstream allRank. See `LICENSE`.

---

allRank is a PyTorch-based framework for training neural Learning-to-Rank (LTR) models, featuring implementations of:
* common pointwise, pairwise and listwise loss functions
* fully connected and Transformer-like scoring functions
* commonly used evaluation metrics like Normalized Discounted Cumulative Gain (NDCG) and Mean Reciprocal Rank (MRR)
* click-models for experiments on simulated click-through data

### Motivation

allRank provides an easy and flexible way to experiment with various LTR neural network models and loss functions.
It is easy to add a custom loss, and to configure the model and the training procedure.
We hope that allRank will facilitate both research in neural LTR and its industrial applications.

## Features

### Implemented loss functions:
 1. ListNet (for binary and graded relevance)
 2. ListMLE
 3. RankNet
 4. Ordinal loss
 5. LambdaRank
 6. LambdaLoss
 7. ApproxNDCG
 8. RMSE
 9. NeuralNDCG (introduced in https://arxiv.org/pdf/2102.07831)

### Getting started guide

To help you get started, we provide a ```run_example.sh``` script which generates dummy ranking data in libsvm format and trains
 a Transformer model on the data using provided example ```config.json``` config file. Once you run the script, the dummy data can be found in `dummy_data` directory
 and the results of the experiment in `test_run` directory. To run the example, Docker is required.

### Configuring your model & training

To train your own model, configure your experiment in ```config.json``` file and run

```python allrank/main.py --config_file_name allrank/config.json --run_id <the_name_of_your_experiment> --job_dir <the_place_to_save_results>```

All the hyperparameters of the training procedure: i.e. model defintion, data location, loss and metrics used, training hyperparametrs etc. are controlled
by the ```config.json``` file. We provide a template file ```config_template.json``` where supported attributes, their meaning and possible values are explained.
 Note that following MSLR-WEB30K convention, your libsvm file with training data should be named `train.txt`. You can specify the name of the validation dataset
 (eg. valid or test) in the config. Results will be saved under the path ```<job_dir>/results/<run_id>```

Google Cloud Storage is supported in allRank as a place for data and job results.

### Implementing custom loss functions

To experiment with your own custom loss, you need to implement a function that takes two tensors (model prediction and ground truth) as input
 and put it in the `losses` package, making sure it is exposed on a package level.
To use it in training, simply pass the name (and args, if your loss method has some hyperparameters) of your function in the correct place in the config file:

```
"loss": {
    "name": "yourLoss",
    "args": {
        "arg1": val1,
        "arg2: val2
    }
  }
```

### Applying click-model

To apply a click model you need to first have an allRank model trained.
Next, run:

```python allrank/rank_and_click.py --input-model-path <path_to_the_model_weights_file> --roles <comma_separated_list_of_ds_roles_to_process e.g. train,valid> --config_file_name allrank/config.json --run_id <the_name_of_your_experiment> --job_dir <the_place_to_save_results>```

The model will be used to rank all slates from the dataset specified in config. Next - a click model configured in config will be applied and the resulting click-through dataset will be written under ```<job_dir>/results/<run_id>``` in a libSVM format.
The path to the results directory may then be used as an input for another allRank model training.

## Continuous integration

You should run `scripts/ci.sh` to verify that code passes style guidelines and unit tests.

## Research

This framework was developed to support the research project [Context-Aware Learning to Rank with Self-Attention](https://arxiv.org/abs/2005.10084). If you use allRank in your research, please cite:
```
@article{Pobrotyn2020ContextAwareLT,
  title={Context-Aware Learning to Rank with Self-Attention},
  author={Przemyslaw Pobrotyn and Tomasz Bartczak and Mikolaj Synowiec and Radoslaw Bialobrzeski and Jaroslaw Bojar},
  journal={ArXiv},
  year={2020},
  volume={abs/2005.10084}
}
```
Additionally, if you use the NeuralNDCG loss function, please cite the corresponding work, [NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting](https://arxiv.org/abs/2102.07831):
```
@article{Pobrotyn2021NeuralNDCG,
  title={NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting},
  author={Przemyslaw Pobrotyn and Radoslaw Bialobrzeski},
  journal={ArXiv},
  year={2021},
  volume={abs/2102.07831}
}
```

## License

Apache 2 License
