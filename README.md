# DecompSR

## Introduction

DecompSR [1] is a benchmark designed to disentangle and evaluate core
dimensions of compositional multihop reasoning in spatial tasks, in
particular: systematicity, productivity, familiarity and
overgeneralisation as well as linguistic variation and robustness to
noise.

This repository contains the code used to create the benchmark and the
resulting [DecompSR benchmark
data](data/processed/DecompSR/DecompSR.jsonl).

## Building the `DecompSR_200` data set

The main script for creating the data is
[parameterized_step_game_8relation_equal_prop.py](src/qa/parameterized_step_game_8relation_equal_prop.py). We
used Python 3.13.3, but the precise version should not be important.
The script can be run as follows:

``` bash
cd ~/git/DecompSR/src/qa
python3 -m venv venv
~/venv/bin/activate
pip install -r requirements.txt
make
```

This results in files in [data/raw/qa](data/raw/qa) with test, train
and validation sets. For convenience, the test data set can be
assembled into a single file, thus:

``` bash
cd ~/git/DecompSr/data/processed/DecompSR
make
```

## Building the larger data sets

For `DecompSR_10K`, we used the following parameters in the [Makefile](src/qa/Makefile):

`python3 parameterized_step_game_8relation_equal_prop.py --seed 143 --proportion_size 1250`

For `DecompSR_100K`, the parameters were:

`python3 parameterized_step_game_8relation_equal_prop.py --seed 143 --proportion_size 12500`

We also had to increase the data set sizes from

``` python
train_size_set = [
        50000
    ] * 100
    test_size = 50000
    valid_size = 1000

```

to

``` python
train_size_set = [
    5000000
    ] *100
    test_size = 5000000
    valid_size = 100000

```

in `parameterized_step_game_8relation_equal_prop.py`.

## Varying the data set

If you want to create a similar data set, but with different stories,
just change the random seed specified in the
[Makefile](src/qa/Makefile) and rerun.

If you want to change the story text (perhaps translating into another
language), change the [template.py](src/qa/template.py) and rerun.

## Acknowledgements

We gratefully acknowledge the work of Fangjun Li
(https://github.com/Fangjun-Li/SpatialLM-StepGame) and Shi
(https://github.com/ShiZhengyan/StepGame) on which this work is based.


## References

[1] DecompSR: a dataset for decomposed analyses of compositional
multihop spatial reasoning
