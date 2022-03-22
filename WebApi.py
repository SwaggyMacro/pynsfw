#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 3/16/2022 3:05 PM
# @Author : Swaggy Macro
# @Site :
# @File : WebApi.py
# @Desc : A Flask WebApi For NSFW
# @Software: PyCharm

import sys
import getopt
import os
import io
import json
import base64
from PIL import Image
from flask import Flask, request
from wsgiref.simple_server import make_server
from nsfw import Nsfw


app = Flask(__name__)


@app.route('/nsfw', methods=["POST"])
def get_tasks():
    if request.method == 'POST':
        base64_str = request.form['img']
        img_b64decode = base64.b64decode(base64_str)
        image = io.BytesIO(img_b64decode)
        image = Image.open(image)
        result = json.dumps(nsfw.check(image))
        return result


ip = '0.0.0.0'
port = '54321'

argv = sys.argv[1:]
try:
	opts, args = getopt.getopt(argv, "i:p:", ["ip=","port="])
	for opt, arg in opts:
		if opt in ['-i', '--ip']:
		   ip = arg
		elif opt in ['-p', '--port']:
		   port = arg
except:
	print("Error")

nsfw = Nsfw(os.path.abspath('.') + "/Model/ckpt.h5")
server = make_server(ip, int(port), app)

print('鉴黄接口启动成功....')
print('---> 地址：http://%s:%s/nsfw'%(ip,port))

server.serve_forever()
app.run()