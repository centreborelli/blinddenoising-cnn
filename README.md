# Neural networks experiments for blind visible/infrared denoising

## Dependencies

* Python 3.7, with the following packages
  - numpy
  - pytorch 1.1.0
  - iio
  - imageio
  - natsort
  - progressbar
  - fire
* Tools from [imscript](https://github.com/mnhrdt/imscript) (plambda)

## Training

1. Download the original fusion data (``cd fusion; bash fusion/dl.sh``) and extract each .tar.gz. The fusion directory should look like this:
```bash
$ tree fusion | head
fusion
├── camouflage
│   ├── take_1
│   │   ├── IR
│   │   │   ├── IR_1000.jpg
│   │   │   ├── IR_1001.jpg
│   │   │   ├── IR_1002.jpg
│   │   │   ├── IR_1003.jpg

```

2. Generate the noisy dataset using (for sigma=20)
```bash
$ bash fusion_add_noise.sh 20
```

3. Launch the training (4 networks)
```bash
$ bash train.sh
```

To monitor the training, the weights and a few resulting frames are saved at each epoch to the directory ```models/```.

## Evaluation

The script ``eval.sh`` takes care of running the 4 networks on all frames of the dataset. Results are saved to the directory ```results/```.
