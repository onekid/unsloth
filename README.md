colab一键训练脚本
https://colab.research.google.com/drive/1MfP3vt9YrOkjg70dKPPFvB174PBnaPSB?usp=sharing

unslo本地安装包下载
百度网盘：https://pan.baidu.com/s/17XehOXC2LMbnLnVebV79lQ?pwd=rycn
谷歌网盘：https://drive.google.com/drive/folders/1BhhBWfOSqCqhmpi8M_dq-nn0eMEZxR-I?usp=sharing
训练的模型下载：https://drive.google.com/file/d/1REtJuRGg2dzRLZ8HyEqfJn8oYuClht8P/view?usp=sharing

相关项目
unsloth：https://github.com/unslothai/unsloth
gpt4all：https://gpt4all.io/
triton：https://github.com/openai/triton
llama.cpp：https://github.com/ggerganov/llama.cpp

Windows本地部署条件
1、Windows10/Windows11
2、英伟达卡8G显存、16G内存，安装CUDA12.1、cuDNN8.9，C盘剩余空间20GB、unsloth安装盘S40GB
3、依赖软件：CUDA12.1+cuDNN8.9、Python11.9、Git、Visual Studio 2022、llvm(可选）
4、HuggingFace账号，上传训练数据集

Windows部署步骤
一、下载安装包
1、安装cuda12.1，配置cuDNN8.9
2、安装Visual Studio 2022
3、解压unsloth
4、安装python11
5、安装git
6、设置llvm系统环境变量(可选）

二、安装unsloth
1、使用python11创建虚拟环境
python311\python.exe -m venv venv
2、激活虚拟环境
call venv\scripts\activate.bat
3、安装依赖包
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install deepspeed-0.13.1+unknown-py3-none-any.whl
pip install  triton-2.1.0-cp311-cp311-win_amd64.whl
pip install xformers==0.0.25.post1
4、测试安装是否成功
nvcc  --version
python -m xformers.info
python -m bitsandbytes
5、运行脚本
test-unlora.py   测试微调之前推理
fine-tuning.py   用数据集微调
test-lora.py   测试微调之后推理
save-16bit.py  合并保存模型16位
save-gguf-4bit.py  4位量化gguf格式
若本地运行fine-tuning.py出错，出现gcc.exe无法编译，可以尝试下载llvm-windows-x64.zip解压，在系统环境变量path路径里添加llvm下的bin路径
三、4位量化需要安装llama.cpp，步骤如下：
1、git clone https://github.com/ggerganov/llama.cpp
2、按官方文档编译
mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=ON
3、设置Visual Studio 2022中cmake路径到系统环境变量path里
C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin
C:\Program Files\Microsoft Visual Studio\2022\Professional
4、编译llama.cpp
cmake --build . --config Release
5、如果上面这句编译命令无法执行，需要做以下操作：
复制这个路径下的
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\extras\visual_studio_integration\MSBuildExtensions
4个文件，粘贴到以下目录里
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations
6、编译好以后，把llama.cpp\build\bing\release目录下的所有文件复制到llama.cpp目录下
7、重新运行fine-tuning.py微调保存为