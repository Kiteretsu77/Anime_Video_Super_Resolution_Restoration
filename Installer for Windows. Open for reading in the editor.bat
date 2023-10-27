@rem Tested to work under Windows here are the links to the video. All files are checked, work correctly.
@rem Video test https://terabox.com/s/1y1JaeyHPc2IvGoV-tYBobQ https://wdho.ru/6Rtz
@rem Install python 3.10
@rem Pre-installation steps copy files from TensorRT-8.6.0.12\onnx_graphsurgeon and TensorRT-8.6.0.12\python and place in .\
@rem This installer supports TensorRT-8.6.0.12 for now but you can change that to whatever version you need.
@rem Post-Installation Steps: Download and copy files from C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin and TensorRT\lib to .\venv\Scripts\
@rem This is necessary so as not to add to PATH.
python -m venv venv
call .\venv\Scripts\activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
pip install tensorrt_dispatch-8.6.0-cp310-none-win_amd64.whl
pip install tensorrt_lean-8.6.0-cp310-none-win_amd64.whl
pip install tensorrt-8.6.0-cp310-none-win_amd64.whl
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
call python setup.py install
