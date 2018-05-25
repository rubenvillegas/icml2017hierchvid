#!/bin/bash
cd ./datasets/Human3.6M

wget -O S9_Hourglass.tar.gz https://umich.box.com/shared/static/3bloh41dlznc2i6f4tcwon1xq2x5vz77.gz
wget -O S11_Hourglass.tar.gz https://umich.box.com/shared/static/huvuit4w0c9vz51uvw2wfqg10ngv3owv.gz

tar -xvzf S9_Hourglass.tar.gz
tar -xvzf S11_Hourglass.tar.gz

rm S9_Hourglass.tar.gz
rm S11_Hourglass.tar.gz

cd ../../

