# Factored Neural Representation for Scene Understanding

[[Project Website]](https://geometry.cs.ucl.ac.uk/projects/2023/factorednerf/)
[[Arxiv]](https://arxiv.org/abs/2304.10950)
[[Dataset (6GB)]](https://geometry.cs.ucl.ac.uk/projects/2023/factorednerf/paper_docs/dataset)

![a](./assets/gif/a_train_rgb.gif)
![b](./assets/gif/b_train_rgb.gif)
![c](./assets/gif/c_train_rgb.gif)

---

## Installation:
    
1. cd to the unzip directory

2. build our docker image
````docker build -t factnerf -f Dockerfile . ````

3. download our dataset and put it at $FACTNERF_ROOT/data
````
    $FACTNERF_ROOT/data/SYN
    $FACTNERF_ROOT/data/SYN/sce_a_train
    ...
````


## Run in a Docker container:

````
export FACTNERF_ROOT=$(pwd)

# check if input data exists
ls $FACTNERF_ROOT/data

# set GPU
export CUDA_VISIBLE_DEVICES=0

````
Training
````
cd $FACTNERF_ROOT 
python framework/run_main.py -f configs/SYN/factorednerf/sce_a.yaml --mode train 
````

Rendering
````
#faster rendering using a smaller resolution
python framework/run_main.py -f configs/SYN/factorednerf/sce_a.yaml --mode render_valid_q  -c map__final --dw 4 --fnum 4 

# rendering (no downsampling)
python framework/run_main.py -f configs/SYN/factorednerf/sce_a.yaml --mode render_valid_q  -c map__final --dw 1 

````


## Checkpoints


## Acknowledgement and Licenses
Some codes are adapted from the awesome repositories: [NiceSlam]( 
https://github.com/cvg/nice-slam) and [Neural Scene Graphs](https://github.com/princeton-computational-imaging/neural-scene-graphs). We appreciated their efforts in open-sourcing their implementation. We also thank the authors of [DeformingThings4D](https://github.com/rabbityl/DeformingThings4D) for allowing us to upload our synthetic dataset. Please be aware of all corresponding licenses.


## Citation
````
@misc{wong2023factored,
      title={Factored Neural Representation for Scene Understanding}, 
      author={Yu-Shiang Wong and Niloy J. Mitra},
      year={2023},
      eprint={2304.10950},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
````

