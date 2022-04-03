#!/bin/bash

cd ~/cpp/OCSMO
cp -rf ~/snort_plugin/OCSMO/makefile .
cp -rf ~/snort_plugin/OCSMO/src/*.* ./src
make