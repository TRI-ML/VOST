# AOT+
This is the implementation of AOT+ baseline used in [VOST](https://www.vostdataset.org) dataset. The implementation is drived from [AOT](https://github.com/z-x-yang/AOT).

## Installation
 We provide a Docker file to re-create the environment which was used in our experiments under `$AOT_ROOT/docker/Dockerfile`. You can either configure the environment yourself using the docker file as a guide or build it via:
  ~~~
    cd $AOT_ROOT
    make docker-build
    make docker-start-interactive
  ~~~ 


## Training and evalaution
1. Link the VOST folder in [datasets/VOST](datasets/VOST)

2. To evaluate the pre-trained AOT+ model on the validation set of VOST download it from [here](https://tri-ml-public.s3.amazonaws.com/datasets/aotplus.pth) into the [pretrain_models](pretrain_models) folder and run the following command:

    ~~~
    python tools/eval.py --exp_name aotplus --stage pre_vost --model r50_aotl --dataset vost --split val --gpu_num 8 --ckpt_path pretrain_models/aotplus.pth --ms 1.0 1.1 1.2 0.9 0.8
    ~~~ 
    To compute the metrics please refer to the [evaluation](../evaluation/) folder.

3. To train AOT+ on VOST yourself download the chekpoint pre-trained on static imges and YouTubeVOS from [here](https://tri-ml-public.s3.amazonaws.com/datasets/pre_ytb.pth) into the [pretrain_models](pretrain_models) and run this script:

     ~~~
    sh train_vost.sh
    ~~~ 


## Citations
Please consider citing the related paper(s) in your publications if it helps your research.
```
@inproceedings{tokmakov2023breaking,
  title={Breaking the “Object” in Video Object Segmentation},
  author={Tokmakov, Pavel and Li, Jie and Gaidon, Adrien},
  booktitle={CVPR},
  year={2023}
}

@inproceedings{yang2021aot,
  title={Associating Objects with Transformers for Video Object Segmentation},
  author={Yang, Zongxin and Wei, Yunchao and Yang, Yi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

## License
This project is released under the BSD-3-Clause license. See [LICENSE](LICENSE) for additional details.
