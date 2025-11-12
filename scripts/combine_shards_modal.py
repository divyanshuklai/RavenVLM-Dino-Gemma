import modal 
import h5py as h5
import numpy
from tqdm import tqdm 

CACHE_VOL = modal.Volume.from_name("cache") 

image = (
    modal.Image.debian_slim("3.13")
    .pip_install(
        "h5py",
        "numpy",
        "tqdm",
    )
)

app = modal.App("SHARD-COMBINER")

@app.function(
    image = image,
    volumes={"/cache":CACHE_VOL},
)
def combine_shards():