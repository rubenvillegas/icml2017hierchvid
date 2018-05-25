#!/bin/bash
cd ./datasets/PennAction

wget -O hourglass.tar.gz https://umich.box.com/shared/static/4agdc5ihb8er6pannnzm2fcdzh8xpw18.gz

tar -xvzf hourglass.tar.gz

rm hourglass.tar.gz

cd ../../

