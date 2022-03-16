# Multi objective Protein Engineering

## Setup
```bash
pip3 install -r requirements.txt
```

## Testing
```
pytest -v
```

## Running an Experiment
```bash
python3 scripts/single_xp_runner.py
```


## Basic Coding Style Guide
- use `assert` __very liberally__
  - assert type is inputs and outputs to all functions
  - assert shapes of all vectors, matrics
  - assert all protein lengths are expected
  - catch soo many dumb bugs asap, massive time saver!

- use docstrings, at the top of any function, write
    - a small idiot proof description of what thefunciton
    is supposed to do
    - a list of the arguments, their meaning, types/shapes
    - a list of outputs, their meaning, types/shapes
    - massive time saver for other people

- minimize variable scope, do not use global variables
  - a function whose behaviour changes due to some variable
  outside of the function is a **nightmare** to debug, a real
  time sink that can destroy productivity, don't do it! Minimizing variable scope is a massive time saver!


