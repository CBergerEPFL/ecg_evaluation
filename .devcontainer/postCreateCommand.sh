#!/bin/bash

pdm --pep582 >> ~/.bash_profile
eval "$(pdm --pep582)"

pdm install
