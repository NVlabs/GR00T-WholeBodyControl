#!/bin/bash
image_name=$(cat image_name.txt)
docker push "$@" $image_name 