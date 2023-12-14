# Pipeline (under development)

## Multiple environment setups

There are at least 3 environments that need to be created to manage steps separately:

```shell
conda env create -f environments/suite2p.yml
conda env create -f environments/deepcad.yml
conda env create -f environments/roicat.yml
```

## Steps

The `scripts/example-steps.sh` show examples of steps to follow for a single animal, single session, single plane.

The following demonstrates how to run things in batches.

### 0-makedirs: Prepare directory structure

**Purpose**: this prepares the directory structure that is expected from all scripts, as well as necessary log folders. Generally the folder structure follows:

```
└── SUBJECT
    ├── DATE
    │   └── STEP
    │       └── PLANE
    └── multi-session
        └── PLANE
```

**Task**: 

- Please edit the variables under `# define variables` in `scripts/0-makedirs.sh` before running
- Then run: 

```shell
bash scripts/0-makedirs.sh
```

- Then put the raw TIFF data under `0-raw` folders

**Output**: The following is an example output directory tree from the script:

<details><summary>Click to expand example output</summary>
    
```
└── MS2457                                              
    ├── 20230902                                          
    │   ├── 0-raw                                       
    │   │   ├── plane0      # put your raw TIFF here                                 
    │   │   └── plane1      # put your raw TIFF here    
    │   ├── 1-moco                                  
    │   │   ├── plane0                                      
    │   │   └── plane1                                  
    │   ├── 2-deepcad                                 
    │   │   ├── plane0                                   
    │   │   └── plane1                                  
    │   ├── 3-suite2p                                 
    │   │   ├── plane0
    │   │   └── plane1
    │   └── 4-muse
    │       ├── plane0
    │       └── plane1
    .
    .
    .
    └── multi-session
        ├── plane0
        └── plane1

```

    
Currently only 1 file per folder is expected.
    
You will be reminded at the end to put your raw data in appropriate folders, like this

```    
Please put raw TIF data in these folders, according to their name
/oscar/data/afleisc2/collab/multiday-reg/data/testdir/MS2457/20230902/0-raw/plane0
/oscar/data/afleisc2/collab/multiday-reg/data/testdir/MS2457/20230902/0-raw/plane1
/oscar/data/afleisc2/collab/multiday-reg/data/testdir/MS2457/20230907/0-raw/plane0
/oscar/data/afleisc2/collab/multiday-reg/data/testdir/MS2457/20230907/0-raw/plane1
/oscar/data/afleisc2/collab/multiday-reg/data/testdir/MS2457/20230912/0-raw/plane0
/oscar/data/afleisc2/collab/multiday-reg/data/testdir/MS2457/20230912/0-raw/plane1
```

So that it looks more like this:
    
```
└── MS2457
    ├── 20230902
    │   ├── 0-raw
    │   │   ├── plane0
    │   │   │   └── MS2457-20230902-plane0.tif
    │   │   └── plane1
    │   │       └── MS2457-20230902-plane1.tif
...
```

</details>


### 1-moco: Motion correction

**Purpose**: this uses `suite2p` to perform motion correction before denoising in a CPU environment on Oscar, preferably high number of CPUs and sufficient memory.

**Task**:

- Please edit the variables under `# define variables` in `scripts/1-moco.sh` before running
- Take a look at `config/1-moco.yml` and change if you need to, refer to [`suite2p` registration parameter documentation](https://suite2p.readthedocs.io/en/latest/settings.html#registration-settings) for more information

- Then run:

```shell
sbatch scripts/1-moco.sh
```

- You can check whether your job is run with `myq`, take note of the `ID` column, then you can view the log file under `logs/1-moco/logs-<JOB_IB>.out`.

**Output**: The following is an example output directory tree from the script:


<details><summary>Click to expand example output</summary>
    
```
└── MS2457
    ├── 20230902
    │   ├── 0-raw
    │   │   ├── plane0
    │   │   │   └── MS2457-20230902-plane0.tif
    │   │   └── plane1
    │   │       └── MS2457-20230902-plane1.tif
    │   ├── 1-moco
    │   │   ├── plane0
    │   │   │   ├── MS2457-20230902-plane0_mc.tif
    │   │   │   └── suite2p-moco-only
    │   │   │       ├── data.bin
    │   │   │       ├── data_raw.bin
    │   │   │       ├── ops.npy
    │   │   │       └── reg_tif
    │   │   │           ├── file00000_chan0.tif
    │   │   │           ├── file001000_chan0.tif
    │   │   │           └── file00500_chan0.tif
    │   │   └── plane1
    │   │       ├── MS2457-20230902-plane1_mc.tif
    │   │       └── suite2p-moco-only                                          
    │   │           ├── data.bin                                               
    │   │           ├── data_raw.bin                                           
    │   │           ├── ops.npy                                                
    │   │           └── reg_tif                                                
    │   │               ├── file00000_chan0.tif                                
    │   │               ├── file001000_chan0.tif                               
    │   │               └── file00500_chan0.tif
...
```
    
The file `MS2457-20230902-plane0_mc.tif` is created by concatenating files under `suite2p-moco-only/re_tif` folder.
    
This file is expected to be here for the next step. 
    
Currently only 1 file per folder is expected.
    
</details>


### 2-deepcad: Denoising using `deepcad`

**Purpose**: this uses `deepcad` to perform denoising on the motion-corrected file, in a GPU environment on Oscar, preferably high memory.

**Task**:

- Please edit the variables under `# define variables` in `scripts/2-deepcad.sh` before running
- Take a look at `config/2-deepcad.yml` and change if you need to
    - [ ] **TODO** Where is documentation for `deepcad` parameters?
- Then run:

```shell
sbatch scripts/2-deepcad.sh
```

- You can check whether your job is run with `myq`, take note of the `ID` column, then you can view the log file under `logs/2-deepcad/logs-<JOB_IB>.out`.

**Output**: The following is an example output directory tree from the script:

<details><summary>Click to expand example output</summary>
    
```
# some folders and files are hidden to prioritize changes
└── MS2457
    ├── 20230902
    │   ├── 0-raw
    │   │   ├── plane0
    │   │   │   └── MS2457-20230902-plane0.tif
    │   │   └── plane1
    │   │       └── MS2457-20230902-plane1.tif
    │   ├── 1-moco
    │   │   ├── plane0
    │   │   │   └── MS2457-20230902-plane0_mc.tif
    │   │   └── plane1
    │   │       └── MS2457-20230902-plane1_mc.tif
    │   ├── 2-deepcad                                                          
    │   │   ├── plane0                                                         
    │   │   │   ├── MS2457-20230902-plane0_mc-dc.tif
    │   │   │   └── para.yaml
    │   │   └── plane1
    │   │       ├── MS2457-20230902-plane1_mc-dc.tif
    │   │       └── para.yaml
...
```

    
Notice files under `2-deepcad` are created, for example `MS2457-20230902-plane0_mc-dc.tif`.

This file is expected to be here for the next step. 
    
</details>


### 3-suite2p: `suite2p` on denoised data

**Purpose**: this uses `suite2p` on the denoised data from the `deepcad` step, in a CPU environment on Oscar, preferably high number of CPUs and sufficient memory.

**Task**:

- Please edit the variables under `# define variables` in `scripts/3-suite2p.sh` before running
- Take a look at `config/3-suite2p.yml` and change if you need to, refer to [`suite2p` parameter documentation](https://suite2p.readthedocs.io/en/latest/settings.html)
    - Currently we still need to re-run registration
    - Some hacks in parameters are documented in this config file
    - If you're familiar with `suite2p`, use your experience and judgements to determine these
- Then run:

```shell
sbatch scripts/3-suite2p.sh
```

- You can check whether your job is run with `myq`, take note of the `ID` column, then you can view the log file under `logs/3-suite2p/logs-<JOB_IB>.out`.

**Output**: The following is an example output directory tree from the script:


<details><summary>Click to expand example output</summary>
    
        
```
└── MS2457
    ├── 20230902
    │   ├── 0-raw
    │   │   ├── plane0
    │   │   │   └── MS2457-20230902-plane0.tif
    │   │   └── plane1
    │   │       └── MS2457-20230902-plane1.tif
    │   ├── 1-moco
    │   │   ├── plane0
    │   │   │   └── MS2457-20230902-plane0_mc.tif
    │   │   └── plane1
    │   │       └── MS2457-20230902-plane1_mc.tif
    │   ├── 2-deepcad                                                          
    │   │   ├── plane0                                                         
    │   │   │   └── MS2457-20230902-plane0_mc-dc.tif
    │   │   └── plane1
    │   │       └── MS2457-20230902-plane1_mc-dc.tif
    │   ├── 3-suite2p                              
    │   │   ├── plane0
    │   │   │   ├── F.npy
    │   │   │   ├── Fneu.npy
    │   │   │   ├── data.bin
    │   │   │   ├── figures
    │   │   │   │   ├── backgrounds.png
    │   │   │   │   └── roi-masks.png
    │   │   │   ├── iscell.npy
    │   │   │   ├── ops.npy
    │   │   │   ├── spks.npy
    │   │   │   └── stat.npy
    │   │   └── plane1
    │   │       ├── F.npy
    │   │       ├── Fneu.npy
    │   │       ├── data.bin
    │   │       ├── figures
    │   │       │   ├── backgrounds.png
    │   │       │   └── roi-masks.png
    │   │       ├── iscell.npy
    │   │       ├── ops.npy
    │   │       ├── spks.npy
    │   │       └── stat.npy
...
```
    
The folders under `3-suite2p` will be very familiar in terms of namings and organizations for `suite2p`-familiar folks
    
</details>

### 4-roicat: Multisession registration using `roicat`

TBD
