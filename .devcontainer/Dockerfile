# 使用官方python镜像为基础
FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# 安装 PyTorch (CPU版) + 其他依赖
RUN pip install --no-cache-dir torch==1.13.1+cpu torchvision==0.14.1+cpu \
    torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

# 安装torchdiffeq本身(也可以在容器跑起来后pip install -e .)
# 这里只演示用pip直接安装最新版本
RUN pip install --no-cache-dir torchdiffeq

# 安装测试/常用包
RUN pip install --no-cache-dir pytest jupyter matplotlib

# 创建一个非root用户(可选, 好处是跟VS Code默认一致)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $yixinnn
WORKDIR /workspace
