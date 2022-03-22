# pynsfw

## 前言

* 训练模型下载：[阿里云盘](https://www.aliyundrive.com/s/nsf99boD8zK) 提取码: `co32` / [百度网盘](https://pan.baidu.com/s/12ITLwNaLuwOEwKqRLOSc-A?pwd=cr27) 提取码: `cr27`
* whl离线依赖：[阿里云盘](https://www.aliyundrive.com/s/ZvV7UV9YfUK) 提取码: `co32` / [百度网盘](https://pan.baidu.com/s/1wai8ufqWIGJR-mqhlOIYQg?pwd=p4ga) 提取码: `p4ga`

## 使用说明

### 快速入门

首先安装项目依赖，都已经生成好了。
直接在项目根目录执行：

```bash
pip install -r requirements.txt
```

运行 `WebApi.py` 文件会通过 `Flask` 构建一个 `WebAPI` 出来，地址是 `0.0.0.0:54321` ，端口号和地址可以自己更改。

```bash
# http://0.0.0.0:54321/nsfw
python WebApi.py

# http://127.0.0.1:2333/nsfw
python WebApi.py -i 127.0.0.1 -p 2333

# http://127.0.0.1:2333/nsfw
python WebApi.py -ip=127.0.0.1 --port=2333
```

`API` 调用方式：直接发起 `Http` 请求，唯一参数 `img` ，参数内容就是 `base64` 后的图片数据（不要开头的文件类型 `data:image/jpg;base64,` ）

返回结果示例(`ret` 就是几率最高的分类名称)：

```json
{
    "ret":"neutral",
    "drawings":"0.0003323109",
    "hentai":"6.302585e-06",
    "neutral":"0.97176874",
    "porn":"0.00476053",
    "sexy":"0.02313208",
    "time":140
}
```

### 使用 Docker 部署

> 注：
> 
> `Dockerfile` 为在线编译
> 
> `Dockerfile-offline` 为离线编译。使用 `Dockerfile-offline` 时，需要将文件名改为 `Dockerfile`，并下载 `Pythone` 离线依赖。[下载地址](https://www.aliyundrive.com/s/ZvV7UV9YfUK) 提取码: `co32`


目录树：

```
pynsfw
 ├── Dockerfile
 ├── Dockerfile-offline
 ├── Model
 │   └── ckpt.h5
 ├── nsfw.py
 ├── README.md
 ├── requirements.txt
 ├── run.sh
 ├── WebApi.py
 └── whl
     ├── absl_py-1.0.0-py3-none-any.whl
     ├── astunparse-1.6.3-py2.py3-none-any.whl
     └── 省略一大堆依赖包......
```

打包编译指令：

```bash
# 下载源码
git clone https://github.com/SwaggyMacro/pynsfw.git

# 进入文件夹
cd pynsfw

# 编译镜像
docker build . -t pynsfw:latest

# 运行
docker run -itd --network=host --name=nsfw nsfw:latest
```




