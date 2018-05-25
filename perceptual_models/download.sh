#!/bin/bash
cd ./perceptual_models

wget -O hourglass.tar.gz https://umich.box.com/shared/static/dhamhcgw57h9anotvhwfg940lcmgskpj.gz
wget -O alexnet.tar.gz https://umich.box.com/shared/static/6dvzphk83f5pqu4hjug48olthqztkn7w.gz

tar -xvzf hourglass.tar.gz
tar -xvzf alexnet.tar.gz

rm hourglass.tar.gz
rm alexnet.tar.gz

cd ../

