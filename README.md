# Code for "Robust Action Segmentation from Timestamp Supervision" - BMVC 2022



Official implementation of [Robust Action Segmentation from Timestamp Supervision](https://arxiv.org/abs/2210.06501).

```
@inproceedings{robust_seg2022,
	title        = {{Robust Action Segmentation from Timestamp Supervision}},
	author       = {Souri, Yaser and Abu Farha, Yazan and Bahrami, Emad and Francesca, Gianpiero and Gall, Juergen},
	year         = 2022,
	booktitle    = {{BMVC}}
}
```



Major parts of the code is adapted from [1].


## Requirements

 - Python >= 3.7 
 - CUDA GPU

Other python requirements are specified in the `requirements.txt` file.

## Data

Download the data from <https://zenodo.org/record/3625992#.Xiv9jGhKhPY> and extract it into `data/`
at the root of the repository.
This is the data provided by [1].


## Running the Experiments

Below is an example of how to run the experiment.
One needs to adjust the arguments to the script for different dataset, splits, and
the amount of timestamp annotations.

```shell
python src/main.py \
    dataset=50salads \
    split=1 \
    timestamp_percentage=90
```

### Parameter β

The parameter β from the paper can be specified in the code by setting the `pgt_config.loss_mul_empty`
argument. For example:

```shell
python src/main.py \
    dataset=50salads \
    split=1 \
    timestamp_percentage=90 \
    pgt_config.loss_mul_empty=0.5
```

### Running the [1] baseline
The type of the pseudo ground truth should be set by `pgt_type=baseline`.

```shell
python src/main.py \
    dataset=50salads \
    split=1 \
    timestamp_percentage=90 \
    pgt_type=baseline
```

### Running the Oracle experiment
The type of the pseudo ground truth should be set by `pgt_type=oracle`.

```shell
python src/main.py \
    dataset=50salads \
    split=1 \
    timestamp_percentage=90 \
    pgt_type=oracle
```



## References
[1] Temporal Action Segmentation from Timestamp Supervision   
Zhe Li, Yazan Abu Farha, Juergen Gall   
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021   
<https://github.com/ZheLi2020/TimestampActionSeg>
