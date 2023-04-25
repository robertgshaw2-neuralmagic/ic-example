## EfficientNet Example

In this example, we will train efficientNet-b1 model on the Flowers102 dataset using the SparseML CLI. We will then quantize the model using QAT and export to ONNX for deployment with DeepSparse.

In this case, we will use the SparseML CLI to train the models. There is also an [example](efficientnet-sparseml-example.ipynb) provided with PyTorch code.

Make sure you have SparseML installed:

```
pip install sparseml[torchvision]
```

## Download the Dataset

We can pass the [Flowers102](https://pytorch.org/vision/main/generated/torchvision.datasets.Flowers102.html#:~:text=Oxford%20102%20Flower%20is%20an,scale%2C%20pose%20and%20light%20variations.) dataset. We need to convert to the ImageFolder format. You can use the following command to download a zip file with the data in the right format.

```bash
pip install gdown
gdown 1vc5KpC2xRxo7ClQIm_pp6yAH9oN2-w8Z
tar -xvf flowers-dataset.tar.gz
```

## Train a Dense Model

Create a folder to run the experiment:

```bash
mkdir dense-run
cd dense-run
```

Run the following to train the dense model:

```bash
sparseml.image_classification.train \
    --checkpoint-path zoo:cv/classification/efficientnet-b1/pytorch/sparseml/imagenet/base-none  \
    --recipe ../recipe.dense.yaml \
    --arch-key efficientnet-b1 \
    --dataset-path ../flowers-dataset \
    --batch-size 16 \
    --train-crop-size 240 \
    --val-crop-size 240 \
    --val-resize-size 256 \
    --weight-decay 0.00001 \
    --interpolation bicubic \
    --random-erase 0.1 \
    --label-smoothing 0.1 \
    --opt rmsprop \
    --auto-augment ta_wide \
    --norm-weight-decay 0.0 \
    --gradient-accum-steps 4
```

The model achieves 99.5% accuracy on the validation set. 

## Quantize the Model

Create a folder to run the quantization experiment:

```bash
cd ..
mkdir quant-run
cd quant-run
```

Run the following to train the model:

```bash
sparseml.image_classification.train \
    --checkpoint-path ../dense-run/checkpoint-best.pth  \
    --recipe ../recipe.quant.yaml \
    --arch-key efficientnet-b1 \
    --dataset-path ../flowers-dataset \
    --batch-size 16 \
    --train-crop-size 240 \
    --val-crop-size 240 \
    --val-resize-size 256 \
    --weight-decay 0.00001 \
    --interpolation bicubic \
    --random-erase 0.1 \
    --label-smoothing 0.1 \
    --opt rmsprop \
    --auto-augment ta_wide \
    --norm-weight-decay 0.0 \
    --gradient-accum-steps 4
```

The quantized model achieves 98.9% accuracy on the validation set.

Note that the CLI produces tensorboard logs by default, so you can monitor the run with tensorboard.

## Export To ONNX

Export the model to ONNX.

```bash
sparseml.image_classification.export_onnx \
    --arch_key efficientnet-b1 \
    --checkpoint_path checkpoint.pth \
    --dataset_path ../flowers-dataset/ \
    --img-resize-size 256 \
    --img-crop-size 240
```

## Run With DeepSparse

Checkout the [example notebook](efficientnet-deepsparse-example.ipynb) showing DeepSparse running the exported model.
