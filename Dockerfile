FROM python:3.9.8

ENV IP=0.0.0.0
ENV PORT=54321

# 创建文件夹
RUN mkdir /nsfw

# 拷贝文件
COPY *.sh /nsfw/
COPY *.py /nsfw/
COPY requirements.txt /nsfw/requirements.txt
COPY Model /nsfw/Model

# 指定工作空间
WORKDIR /nsfw

# 安装依赖
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt && \
	chmod +x run.sh

ENTRYPOINT ["/nsfw/run.sh"]