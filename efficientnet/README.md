### Train a Dense Model

Create a folder to run the experiment:

```bash
mkdir dense-run
cd dense-run
```

Run the following to train the dense model:

```bash
sparseml.image_classification.train --checkpoint-path zoo:cv/classification/efficientnet-b1/pytorch/sparseml/imagenet/base-none  --recipe ../recipe.dense.yaml --arch-key efficientnet-b1 --dataset-path ../flowers-dataset --batch-size 16 --train-crop-size 240 --val-crop-size 240 --val-resize-size 256 --weight-decay 0.00001 --interpolation bicubic --random-erase 0.1 --label-smoothing 0.1 --opt rmsprop --auto-augment ta_wide --norm-weight-decay 0.0 --gradient-accum-steps 4
```

### Quantize the Model

Create a folder to run the quantization experiment:

```bash
cd ..
mkdir quant-run
cd quant-run
```

Run the following to train the model:

```bash
sparseml.image_classification.train --checkpoint-path ../dense-run/checkpoint-best.pth  --recipe ../recipe.quant.yaml --arch-key efficientnet-b1 --dataset-path ../flowers-dataset --batch-size 16 --train-crop-size 240 --val-crop-size 240 --val-resize-size 256 --weight-decay 0.00001 --interpolation bicubic --random-erase 0.1 --label-smoothing 0.1 --opt rmsprop --auto-augment ta_wide --norm-weight-decay 0.0 --gradient-accum-steps 4
```

Export the model to ONNX.

```
sparseml.image_classification.export_onnx --arch_key efficientnet-b1 --checkpoint_path checkpoint.pth --dataset_path ../flowers-dataset/ --img-resize-size 256 --img-crop-size 240
```