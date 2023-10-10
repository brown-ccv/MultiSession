# Pipeline for cell registration across multiple sessions

## Environment configuration

This package is intended for use on Oscar with GPU support for PyTorch.

To use on Oscar, first clone the repo. Then load the anaconda module:

```shell
module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
```

If you have never loaded the `anaconda/2022.05` module before, you need to initialize
the conda module by running the following command.

```shell
conda init bash
```

For more information on working with conda on Oscar, please consult [this page](https://docs.ccv.brown.edu/oscar/software/anaconda).

Once the anaconda module is loaded, please create a new environment using the `environment.yml` file:

```shell
conda env create -n cellreg -f environment.yml
```

where `cellreg` is the name of your environment. Please be patient as conda is slow on
Oscar. Once the environment has been created, activate the environment with

```shell
conda activate cellreg
```
