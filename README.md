### Data Preparation

- Put the tracking datasets in `./data`. It should look like this:

```
${PROJECT_ROOT}
 -- data
     -- lasot
         |-- airplane
         |-- basketball
         |-- bear
         ...
     -- got10k
         |-- test
         |-- train
         |-- val
     -- coco
         |-- annotations
         |-- images
     -- trackingnet
         |-- TRAIN_0
         |-- TRAIN_1
         ...
         |-- TRAIN_11
         |-- TEST
```

- Edit the dataset path in  `./consts.yaml`

  ```yaml
  LaSOT_PATH: '/path/to/lasot'
  GOT10k_PATH: '/path/to/got10k'
  TrackingNet_PATH: '/path/to/trackingnet'
  COCO_2017_PATH: '/path/to/coco2017'
  ```

  

### Install the environment

Our implementation is based on PyTorch 2.3.1+CUDA12. Use the following command to install the runtime environment:

```
conda env create -f DMTrack_env_cuda12.yaml
```



### Training

Note: Our code performs evaluation automatically when model training is complete.

- ***Model weight*** is saved in ```/path/to/output/run_id/checkpoint/epoch_{last}/model.bin```.

- ***Performance metrics*** can be found on terminal output and wandb dashboard.

- ***Tracking results*** are saved in ```/path/to/output/run_id/eval/epoch_{last}/```.

For convenience, you can just run `bash train.sh`  to start training DMTrack-B.

##### Using boot.sh helper script

```shell
# Train and evaluate DMTrack model on all GPUs
./boot.sh DMTrack dinov2 --output_dir /path/to/output
# Train and evaluate DMTrack-B model following GOT-10k protocol on all GPUs
./boot.sh DMTrack dinov2 --output_dir /path/to/output --mixin got10k
# Train and evaluate on specific GPUs
./boot.sh DMTrack dinov2 --output_dir /path/to/output --device_ids 0,1,2,3
# Train and evaluate on multiple nodes
./boot.sh DMTrack dinov2 --output_dir /path/to/output --nnodes $num_nodes --node_rank $node_rank --master_address $master_node_ip --date 2025.06.09-12.00.00-123456
```

You can set the default settings, e.g. `output_dir`, in ```boot.sh```.

You may encounter the Nan Loss issue during training. We recommend: First, try more times; second, use the following command to disable torch.compile, which can further improve stability.

```shell
./boot.sh DMTrack dinov2 --output_dir /path/to/output --mixin disable_torch_compile
```

### Evaluation

Our code performs evaluation automatically when model training is complete. You can run evaluation only with the following command:

```shell
# evaluation only, on all datasets, defined in config/_dataset/test.yaml
./boot.sh DMTrack dinov2 --output_dir /path/to/output --mixin evaluation --weight_path /path/to/weight.bin
```

The evaluated datasets are defined in ```config/_dataset/test.yaml```.

#### TrackingNet evaluation

Once the full evaluation is done, result files are saved in ```/path/to/output/run_id/eval/epoch_{last}/TrackingNet-test.zip```.

Submit this file to the [TrackingNet evaluation server](https://tracking-net.org/) to get the result of TrackingNet test split.

### Train and evaluate with GOT-10k dataset

```shell
# Train and evaluate DMTrack-B model following GOT-10k protocol on all GPUs
./boot.sh DMTrack dinov2 --output_dir /path/to/output --mixin got10k
```

Submit ```/path/to/output/run_id/eval/epoch_{last}/GOT10k-test.zip``` to the [GOT-10k evaluation server](http://got-10k.aitestunion.com/) to get the result of GOT-10k test split.

Evaluation only:

```shell
# evaluation only, on GOT-10k dataset
./boot.sh DMTrack dinov2 --output_dir /path/to/output --mixin got10k --mixin evaluation --weight_path /path/to/weight.bin
```

Note that, as defined in ```config/LoRAT/_mixin/got10k.yaml```, we evaluate GOT-10k dataset three times.





## Acknowledgement

Thanks to the following repo for providing us with a lot of convenience to implement our method.

- [LoRAT](https://github.com/LitingLin/LoRAT)
- [SPMTrack](https://github.com/WenRuiCai/SPMTrack)

