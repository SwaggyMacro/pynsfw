#!/bin/bash

if [ -n "${NSFW_OPTIONS}" ]; then
	python WebApi.py ${NSFW_OPTIONS}
else
	python WebApi.py
fi