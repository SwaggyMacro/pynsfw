#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 3/16/2022 3:05 PM
# @Author : Swaggy Macro
# @Site :
# @File : WebApi.py
# @Desc : A Flask WebApi For NSFW
# @Software: PyCharm

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


nsfw = Nsfw(os.path.abspath('.') + "\\Model\\ckpt.h5")
server = make_server('0.0.0.0', 54321, app)
server.serve_forever()
app.run()

