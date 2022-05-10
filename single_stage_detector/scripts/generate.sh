export PYTHONPATH="${PYTHONPATH}:/home/htajalli/github/farshad-t/retina-net-training/single_stage_detector/ssd"
python pth_to_onnx_training.py --device cpu --batch-size 4 --num-classes 264 --num-obj 9 --output resnext50_32x4d_fpn_c264_bs4_training.onnx
python pth_to_onnx_training.py --device cpu --batch-size 8 --num-classes 264 --num-obj 9 --output resnext50_32x4d_fpn_c264_bs8_training.onnx
python pth_to_onnx_training.py --device cpu --batch-size 14 --num-classes 264 --num-obj 9 --output resnext50_32x4d_fpn_c264_bs14_training.onnx
python pth_to_onnx_training.py --device cpu --batch-size 16 --num-classes 264 --num-obj 9 --output resnext50_32x4d_fpn_c264_bs16_training.onnx
python pth_to_onnx_training.py --device cpu --batch-size 28 --num-classes 264 --num-obj 9 --output resnext50_32x4d_fpn_c264_bs28_training.onnx
python pth_to_onnx_training.py --device cpu --batch-size 32 --num-classes 264 --num-obj 9 --output resnext50_32x4d_fpn_c264_bs32_training.onnx
python pth_to_onnx_training.py --device cpu --batch-size 2 --num-classes 264 --num-obj 9 --output resnext50_32x4d_fpn_c264_bs2_training.onnx
python pth_to_onnx_training.py --device cpu --batch-size 1 --num-classes 264 --num-obj 9 --output resnext50_32x4d_fpn_c264_bs1_training.onnx
