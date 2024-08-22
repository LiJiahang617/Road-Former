# Road-Former
Offical RoadFormer series scene parsing networks implementation based on mmsegmentation v1.0.0. All the related code and datasets will be released upon publication.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/roadformer-delivering-rgb-x-scene-parsing/thermal-image-segmentation-on-mfn-dataset)](https://paperswithcode.com/sota/thermal-image-segmentation-on-mfn-dataset?p=roadformer-delivering-rgb-x-scene-parsing)  

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/roadformer-delivering-rgb-x-scene-parsing/semantic-segmentation-on-fmb-dataset)](https://paperswithcode.com/sota/semantic-segmentation-on-fmb-dataset?p=roadformer-delivering-rgb-x-scene-parsing)  

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/roadformer-delivering-rgb-x-scene-parsing/semantic-segmentation-on-zju-rgb-p)](https://paperswithcode.com/sota/semantic-segmentation-on-zju-rgb-p?p=roadformer-delivering-rgb-x-scene-parsing)  

## News
- [2023/09/16]: Our paper RoadFormer is submitted to the ICRA 2024.
- [2024/01/31]: Unfortunately, RoadFormer was rejected by ICRA 2024 after peer review😭😭😭.
- [2024/03/31]: Our paper RoadFormer has been accepted by the IEEE T-IV 2024. Stay tuned for the RoadFormer+!
- [2024/07/17]: Our series of works, RoadFormer and RoadFormer+, are nearing completion. Upon finalizing the organization of the code, we will release it. Stay tuned for updates!
- [2024/08/21]: Our paper RoadFormer+ has been accepted by the IEEE T-IV 2024. Stay tuned for our code!
- [2024/08/22]: The implementation of our RoadFormer series, has been released, the utilized SYN-UDTIRI will be released soon.

## Citation
If you find our works useful in your research, please consider citing:
```
@article{li2024roadformer,
  title={RoadFormer: Duplex transformer for RGB-normal semantic road scene parsing},
  author={Li, Jiahang and Zhang, Yikang and Yun, Peng and Zhou, Guangliang and Chen, Qijun and Fan, Rui},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2024},
  publisher={IEEE},
  note={{DOI}:{10.1109/TIV.2024.3388726}},
}

@article{huang2024roadformer+,
  title={RoadFormer+: Delivering RGB-X Scene Parsing through Scale-Aware Information Decoupling and Advanced Heterogeneous Feature Fusion},
  author={Huang, Jianxin and Li, Jiahang and Jia, Ning and Sun, Yuxiang and Liu, Chengju and Chen, Qijun and Fan, Rui},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2024},
  publisher={IEEE},
  note={{DOI}:{10.1109/TIV.2024.3448251}},
}
```
## Usage 

### Installation

```bash
git clone https://github.com/LiJiahang617/Road-Former.git
cd Road-Former
```
Please refer to [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation) for the dependency instructions.
You will need a compiled ``.so`` file, or you can download the pre-compiled file from here:
- [_ext.cpython-38-x86_64-linux-gnu.so](https://pan.baidu.com/s/1yg52J4umKiPLVVDeFwLfBA?pwd=apei)

This file is complied using the NVIDIA RTX 3090 GPU, which may cause error when under other devices.
### Running

```bash
python tools/train.py --config <config-file-path>
python tools/test.py --config <config-file-path> --pretrained <pre-trained-pth-path>
```
For more training and inference details, please refer to MMSegmentation [instructions](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/4_train_test.md).

