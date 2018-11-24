#!/usr/bin/env bash

cd CD
sh tf*abi.sh
cd ..

cd emd
sh tf*abi.sh
cd ..

cd grouping
sh tf*abi.sh
cd ..

cd interpolation
sh tf*abi.sh 
cd ..

cd sampling
sh tf*abi.sh
cd ..
