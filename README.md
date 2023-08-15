# PSNE
implementations of the model proposed in the paper:
PSNE: Scaling Structure-aware Network Embedding
via Spectral Sparsification
## Requirements
Start with conda 
```bash
conda create -n your_env_name python=3.8.16
```
Please install dependencies by
```bash
pip install -r requirements.txt
```

## Datasets
- PPI contains 3,890 nodes, 76,584 edges and 50 labels.
- Wikipedia contains 4,777 nodes, 184,812 edges and 40 labels.
- Blogcatalog contains 10,312 nodes, 333,983 edges and 39 labels.
- Flickr contains  80,513 nodes, 5,899,882 edges and 195 labels. 
- Youtube contains 1,138,499 nodes, 2,990,443 edges and 47 labels.
## Run
```bash
python main.py
```
## Hyperparameter modification and choose datasets
All parameters are read from my_file.yaml, you can directly modify the parameters and the url of the dataset in my_file.yaml
## Cite
Please cite our paper if you use the model or this code in your own work:
