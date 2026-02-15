# 🚀 Frequency-Enhanced Hilbert Scanning Mamba for Short-Term Arctic Sea Ice Concentration Prediction, IEEE TGRS 2026

## Model

FH-Mamba for Predicting Sea Ice Concentration (SIC) in the Arctic.

This study utilizes Frequency-enhanced Hilbert scanning Mamba to simulate the spatial-temporal evolution of sea ice concentration. The model incorporates Discrete Wavelet Transform (DWT) for frequency information extraction and Hilbert Scanning 3D Mamba for capturing long-range spatial-temporal dependencies efficiently.

## How to Use

### Environment

Install the required dependencies (such as PyTorch, Mamba-ssm, etc.) to create the python environment.

### Dataset

We recommend downloading and organizing the dataset in the following order to avoid any issues.

#### 1. Download and Reorganize Data

The monthly SIC data used in this study can be downloaded from OSI SAF **OSI-450-a1** (Global Sea Ice Concentration Climate Data Record v3.0), available at https://osi-saf.eumetsat.int/products/osi-450-a1, which also contains a detailed description of the dataset and user guide.

Run the `download.py` file in the `data` directory to download the data.

Run `organize.py` in the `data` directory to reorganize the data.

The reorganized data structure:

```
data
├── 1991
├── 1992
├── 1993
......
├── 2019
│   ├── 01
│   │   ├── ice_conc_nh_ease2-250_cdr-v3p1_201901011200.nc
│   │   ├── ice_conc_nh_ease2-250_cdr-v3p1_201901021200.nc
│   │   ├── ice_conc_nh_ease2-250_cdr-v3p1_201901031200.nc
│   │   ├── ice_conc_nh_ease2-250_cdr-v3p1_201901041200.nc
......
├── 2020
```

#### 2. Generate file containing the path to all the data files

```shell
bash gen_data_text.sh
```

The generated `data.txt` file is located in the `data` directory.

The content of `data.txt` should be as follows:

```
./1991/01/ice_conc_nh_ease2-250_cdr-v3p1_199101011200.nc
./1991/01/ice_conc_nh_ease2-250_cdr-v3p1_199101021200.nc
./1991/01/ice_conc_nh_ease2-250_cdr-v3p1_199101031200.nc
./1991/01/ice_conc_nh_ease2-250_cdr-v3p1_199101041200.nc
......
```

#### 3. Read and store the processed data to facilitate future data loading

Run `full_sic.py`. This will call the `write_netcdf` function, passing the filename from the previous step as an argument, to read and store the processed data. A `full_sic.nc` file will be generated in the `data` directory to facilitate future data reading.

```python
# full_sic.py
# Read and store the processed data for easier loading next time by calling the write_netcdf function 
# and passing the output filename from the previous step as an argument.
from utils import write_netcdf
start_time = 19910101
end_time = 20201231
write_netcdf("data.txt", start_time, end_time, "full_sic.nc")
```

## Train

Change relevant parameters and the `.nc` file path in the `config.py` file, put all the scripts under the same folder and run:

```shell
python train.py
```

The training progress will be printed to the console. You can also choose to redirect this information to a log file.

## Test

Specify the testing period and output directory, then run:

```sh
python test.py -ts 20160101 -te 20161231
python test.py -ts 20170101 -te 20171231
python test.py -ts 20180101 -te 20181231
python test.py -ts 20190101 -te 20191231
python test.py -ts 20200101 -te 20201231
```

Args:

```python
parser.add_argument('-ts', '--start_time', type=int,
                        required=True, help="Starting time (eight digits, YYYYMMDD)")
parser.add_argument('-te', '--end_time', type=int,
                        required=True, help="Ending time (eight digits, YYYYMMDD)")
```