## EfficientNet Example

In this example, we will train efficientNet-b1 model on the Flowers102 dataset using the SparseML CLI. We will then quantize the model using QAT and export to ONNX for deployment with DeepSparse.

In this case, we will use the SparseML CLI to train the models. There is also an [example](efficientnet-sparseml-example.ipynb) provided with PyTorch code.

Make sure you have SparseML installed:

```
pip install sparseml[torchvision]
```
