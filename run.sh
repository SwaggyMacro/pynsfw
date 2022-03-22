#!/bin/bash

echo "=================== 启动信息 ==================="
echo "即将启动鉴黄接口...."
echo "---> 地址：http://${IP}:${PORT}/nsfw"
echo "==============================================="

if [ -n "${IP}" ] && [ -n "${PORT}" ]; then
	python WebApi.py --ip=${IP} --port=${PORT}
else
	python WebApi.py
fi