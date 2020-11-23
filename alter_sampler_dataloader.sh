mv /usr/local/lib/python3.6/site-packages/torch/utils/data/dataloader.py /usr/local/lib/python3.6/site-packages/torch/utils/data/dataloader.py.ori
cp dataloader.py /usr/local/lib/python3.6/site-packages/torch/utils/data/dataloader.py
mv /usr/local/lib/python3.6/site-packages/torch/utils/data/sampler.py /usr/local/lib/python3.6/site-packages/torch/utils/data/sampler.py.ori
cp sampler.py /usr/local/lib/python3.6/site-packages/torch/utils/data/sampler.py
vim /usr/local/lib/python3.6/site-packages/torch/utils/data/__init__.py
