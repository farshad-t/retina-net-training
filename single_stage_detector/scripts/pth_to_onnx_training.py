#!/usr/bin/env python3
import argparse

import torch
import torch.onnx
import torchvision
from torch.autograd import Variable

from model.retinanet import retinanet_from_backbone, RetinaNet,\
    RetinaNetClassificationHead, RetinaNetHead, RetinaNetRegressionHead,\
    RetinaNetClassificationHeadLoss, RetinaNetRegressionHeadLoss, IOU

import torch._C as _C
from distutils.command.check import check
from model.anchor_utils import AnchorGenerator

TrainingMode = _C._onnx.TrainingMode
OperatorExportTypes = _C._onnx.OperatorExportTypes

def parse_args(add_help=True):
    parser = argparse.ArgumentParser(description='Convert PyTorch detection file to onnx format', add_help=add_help)

    parser.add_argument('--input', required=False, help='input pth file')
    parser.add_argument('--output', default=None, help='output onnx file')

    parser.add_argument('--backbone', default='resnext50_32x4d',
                        choices=['resnet50', 'resnext50_32x4d', 'resnet101', 'resnext101_32x8d'],
                        help='The model backbone')
    parser.add_argument('--num-classes', default=264, type=int,
                        help='Number of detection classes')
    parser.add_argument('--trainable-backbone-layers', default=3, type=int,
                        help='number of trainable layers of backbone')

    parser.add_argument('--image-size', default=None, nargs=2, type=int,
                        help='Image size for training. If not set then will be dynamic')
    parser.add_argument('--batch-size', default=None, type=int,
                        help='input batch size. if not set then will be dynamic')
    parser.add_argument('--num-obj', default=10, type=int,
                        help='average num-objects found in training images.')
    parser.add_argument('--data-layout', default="channels_first", choices=['channels_first', 'channels_last'],
                        help="Model data layout")
    parser.add_argument('--device', default='cuda', help='device')

    args = parser.parse_args()

    args.output = args.output or (f'/home/parallels/github/farshad-t/frameworks.ai.benchmarking.archbench-1/networks/WIP_networks/{args.backbone}_fpn_c{args.num_classes}_bs{args.batch_size}_training.onnx')
    return args

def main(args):
    batch_size = args.batch_size or 1
    image_size = args.image_size or [800, 800]
    nc = args.num_classes

    print("Creating model")
    model = retinanet_from_backbone(backbone=args.backbone,
                                    num_classes=args.num_classes,
                                    image_size=image_size,
                                    data_layout=args.data_layout,
                                    pretrained=False,
                                    trainable_backbone_layers=args.trainable_backbone_layers)
    device = torch.device(args.device)
    model.to(device)

    #print("Loading model")
    #checkpoint = torch.load(args.input)
    #model.load_state_dict(checkpoint['model'])

    print("Creating input tensor")

    inputs = torch.randn(batch_size, 3, image_size[0], image_size[1],
               device=device,
               requires_grad=False,
               dtype=torch.float)


    # Output dynamic axes
    dynamic_axes = {
        'boxes': {0 : 'num_detections'},
        'scores': {0 : 'num_detections'},
        'labels': {0 : 'num_detections'},
    }
    # Input dynamic axes
    if (batch_size is None) or (image_size is None):
        dynamic_axes['images'] = {}
        if batch_size is None:
            dynamic_axes['images'][0]: 'batch_size'
        if image_size is None:
            dynamic_axes['images'][2] = 'width'
            dynamic_axes['images'][3] = 'height'
    
    targets = []
    no = args.num_obj
    const_base = torch.tensor([0,0,image_size[0]/2,image_size[0]/2])
    constant_tensor = const_base * torch.ones(batch_size, no, 4)
    targets.append({'boxes':torch.randint(batch_size, image_size[0]//2,(batch_size, no, 4)) + constant_tensor, 
                    'labels':torch.randint(nc,(batch_size, no,))})

    print("Exporting the model:"+args.output)
    torch.onnx.export(model,
                      (inputs, targets),
                      #(inputs, [{'boxes':torch.randn(0,4), 'labels':[torch.randn(0)]}]),
                      args.output,
                      export_params=True, #export trained params
                      opset_version=15,
                      do_constant_folding=True,
                      input_names=['images', 'targets'],
                      output_names=['losses', 'detections'],
                      dynamic_axes=dynamic_axes,
                      training=TrainingMode.TRAINING,
                      operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
                      keep_initializers_as_inputs=False,
                      export_modules_as_functions={RetinaNetRegressionHeadLoss, AnchorGenerator, RetinaNetClassificationHeadLoss, IOU})
 

#                       operator_export_type=OperatorExportTypes.ONNX)
#                       operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH)
#                       operator_export_type=OperatorExportTypes.ONNX_ATEN)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("done")
