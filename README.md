<!--
SPDX-FileCopyrightText: None

SPDX-License-Identifier: CC0-1.0
-->
# Mein eigener Synthesizer

## Install 

Set up a virtual environment (optional) and install dependencies:

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements
```

## Set-up and run 

List available sound devices:

```shell
python3 ./mein-eigener-synthesizer.py --list-devices
```

Run with preferred device (using `--device` option)

```shell
python3 ./mein-eigener-synthesizer.py --device 0 # put your preferred device here
```

Example with different sounds
```shell
python3 ./mein-eigener-synthesizer.py \
  --base G \
  --octave 3 \
  --instrument beam \
  --excitation hard \
  --pickup clean \
  --duration 2
```


## What is this about?

Exploring eigenfunctions and eigenvalues of differential operators by making 
sounds from them.