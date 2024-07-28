colab一键训练脚本
https://colab.research.google.com/drive/1MfP3vt9YrOkjg70dKPPFvB174PBnaPSB?usp=sharing

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
若本地运行fine-tuning.py出错，出现gcc.exe无法编译，可以尝试下载llvm-windows-x64.zip解压，在系统环境变量path路径里添加llvm下的bin路径
三、4位量化需要安装llama.cpp，步骤如下：
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make GGML_CUDA=1（没有gpu的linux使用make）

# obtain the official LLaMA model weights and place them in ./models
ls ./models
llama-2-7b tokenizer_checklist.chk tokenizer.model
# [Optional] for models using BPE tokenizers
ls ./models
<folder containing weights and tokenizer json> vocab.json
# [Optional] for PyTorch .bin models like Mistral-7B
ls ./models
<folder containing weights and tokenizer json>

# install Python dependencies
python3 -m pip install -r requirements.txt

# 转换模型为ggml FP16格式（cd ./llama.cpp)
python convert-hf-to-gguf.py ../outputs    --outfile ./mymodel/namemv my	.gguf --outtype f16

# 四位量化 (using Q4_K_M method)（cd  ./llama.cpp)
./llama-quantize ./mymodel/ggml-model-f16.gguf ./mymodel/ggml-model-Q4_K_M.gguf Q4_K_M

# update the gguf filetype to current version if older version is now unsupported
./llama-quantize ./models/mymodel/ggml-model-Q4_K_M.gguf ./models/mymodel/ggml-model-Q4_K_M-v2.gguf COPY

#直接使用模型
./llama-cli -m ./models/mymodel/ggml-model-Q4_K_M.gguf -n 128

交互模式：
# default arguments using a 7B model
./examples/chat.sh

# advanced chat with a 13B model
./examples/chat-13B.sh

# custom arguments using a 13B model
./llama-cli -m ./models/13B/ggml-model-q4_0.gguf -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
