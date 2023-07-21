
This instruction is for people in FAIR (Meta AI) to run experiments.


## Installation on devfair

Run `echo $CC` to verify the environment variables related to compilers are set. It should output a path of the clang compiler.

Follow these steps to set up a development environment on devfair:

1. **Setup conda environment:**

```sh
conda create -n rlcompopt python=3.8 cmake pandoc patchelf
conda activate rlcompopt
```

2. **Install bazel:** Bazel is used to compile the C++/python package. Here we
   will use bazelisk to manage our bazel installation and download it to
   `~/.local/bin`:

```sh
mkdir -p ~/.local/bin
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.7.5/bazelisk-linux-amd64 -O bazel
chmod +x bazel && mkdir -p ~/.local/bin && mv -v bazel ~/.local/bin
export PATH=~/.local/bin:$PATH
```

3. **Install PyTorch:** The codebase requires 2.0 > PyTorch >= 1.12.1. We can install it following [here](https://pytorch.org/get-started/previous-versions). We recommend using conda to install PyTorch to avoid possible dependencies conflict. You need to find the correct command according to the CUDA version your GPU driver supports (check `nvidia-smi`). For example, I found my GPU driver supported CUDA 11.6, so I run `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia` to install pytorch 1.13.1. After the installation, verify PyTorch is usable on GPU by running `python -c "import torch; print(torch.matmul(torch.rand(2, 8).to(0), torch.rand(8, 4).to(0)).shape)"`. If it outputs `torch.Size([2, 4])` then we can go to next step, otherwise try to fix the issues by reinstall PyTorch.


4. **Install `torch-geometric`, `pyzmq`, and logging tools:** 
We recommend using conda to install `torch-geometric` and `pyzmq` to avoid possible dependencies conflict. 
```sh
conda install -c pyg pyg=2.1.0
conda install -c anaconda pyzmq=23.2.0
conda install -c dglteam dgl=1.1.0
cd ..
git clone https://github.com/yuandong-tian/tools2.git
cd tools2
python -m pip install .
```

5. **Clone CompilerGym and this repo:** We will check out both this repo and
   CompilerGym and install all development dependencies by running the following commands. Note that we clone the specific folk of CompilerGym that includes the type graph patch. We change to a desired directory to clone the repo: `cd /path/of/your/choice`.


```sh
cd ..
git clone --depth 1 --branch rlcompopt https://github.com/youweiliang/CompilerGym.git
cd CompilerGym
make init

cd ..
git clone https://github.com/facebookresearch/RLCompOpt.git
cd RLCompOpt
make init
```

6. **Build and install CompilerGym from source.** 

Run `pip install setuptools==65.5.0` so that the library [gym](https://github.com/openai/gym) can be installed properly.

```sh
cd ../CompilerGym
make install
```

If you want to modify the CompilerGym codebase, you need to make your desired changes and then re-run `make install`.


7. **Install this repo:**

```sh
cd ../RLCompOpt
make install
```
**If you modify this repo, you will need to reinstall it to make any changes to take effect.**

8. **Use RAM rather than NFS for faster environments:** CompilerGym
   does quite a lot of disk operations which can be slow on the cluster NFS.
   Force CompilerGym to instead keep everything in memory using:

```sh
export COMPILER_GYM_SITE_DATA=/dev/shm/compiler_gym_site_data
```


9. (Optional) **Automate the environment setup:** Create a script to set up
   these environment variables so that you don't have to redo it next time you
   spawn a shell:

```sh
cat <<EOF > ~/.rlcompopt_env
conda activate rlcompopt
export PATH=$HOME/.local/bin:$PATH
export COMPILER_GYM_SITE_DATA=/dev/shm/compiler_gym_site_data

EOF
```

Now you can do `source ~/.rlcompopt_env` to restore the environment.


## Preparing data files
The data files can be downloaded from this [Google Drive](https://drive.google.com/drive/folders/1lATNWBKmsubw8bGeFyDlBHXlYbcRrw7S?usp=sharing). You can install gdown to download it:
```
conda install -c conda-forge gdown
gdown --folder https://drive.google.com/drive/folders/1lATNWBKmsubw8bGeFyDlBHXlYbcRrw7S?usp=sharing
```
The commands should save the files under a folder named data. 

Or you can download it from the website and place the data folder under the repo, which results in the following file structure.
```
data
├── all_ssl_vocab.db
...
```

## Training

**Note that you may need to load CUDA/CUDNN modules**.

### Training and testing of Normalized Value Prediction (NVP), Behavior Cloning (BC), and Q value
Same as the instructions in README. If you submit the training jobs to Slurm, the testing script will be automatically submitted to Slurm once the training is done. So you don't need to start testing manually.


### Training and testing RL-PPO agents
Same as the instructions in README.

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
RLCompOpt is MIT licensed, as found in the [LICENSE](LICENSE) file.

## Citing RLCompOpt
```BibTeX
@InProceedings{liang2023rlcompopt,
  title={Learning Compiler Pass Orders using Coreset and Normalized Value Prediction},
  author={Liang, Youwei and Stone, Kevin and Shameli, Ali and Cummins, Chris and Elhoushi, Mostafa and Guo, Jiadong and Steiner, Benoit and Yang, Xiaomeng and Xie, Pengtao and Leather, Hugh and Tian, Yuandong},
  year={2023},
  booktitle={Proceedings of the 40th International Conference on Machine Learning}
}
```