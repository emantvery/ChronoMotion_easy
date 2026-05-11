# 使用Python 3.8基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir flask==2.3.0

# 暴露端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/v1/health || exit 1

# 启动命令
CMD ["python", "demo_app.py"]