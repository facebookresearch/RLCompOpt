
This repo contains experiments to learn to optimize program compilation using RL.

For people in FAIR (Meta AI), check [README_FAIR.md](README_FAIR.md) to get started.

## System requirements
The codebase was tested on Ubuntu 18.04. To install some possible missing libraries on Ubuntu 18.04, we need to run `sudo apt-get install libtinfo-dev` and `sudo apt-get install m4`. 


## Installing compilers

We use `~/.local/opt` as the installation directory of compilers.

```sh
# Download and unpack a modern clang release.
mkdir -p ~/.local/opt && cd ~/.local/opt
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar xf clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
```

We then need to set some environment variables whenever we build or use
CompilerGym. The easiest way to do that is to add them to your `~/.bashrc`:

```sh
cat <<EOF >>~/.bashrc
# === Building CompilerGym ===

# Set clang as the compiler of choice.
export CC=$HOME/.local/opt/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/clang
export CXX=$HOME/.local/opt/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/clang++
export PATH=$HOME/.local/opt/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin:$PATH
export BAZEL_BUILD_OPTS=--repo_env=CC=$HOME/.local/opt/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/clang
EOF
```

So the environment variables are set every time you logs in, or you can run `source ~/.bashrc` in the current shell to set the environment variables. Run `echo $CC` to verify the environment variables are set. It should output a path of the clang compiler.

## Environment setup

Follow these steps to set up a development environment on Ubuntu 18.04 (or any other Linux
/ macOS machine, with some tweaks). 

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

```sh
cd ../CompilerGym
make install
```
If you encounter an error related to installing the library [gym](https://github.com/openai/gym), try to run `pip install setuptools==65.5.0` and then run `make install` again (see this [issue](https://github.com/openai/gym/issues/3176)).

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

(Optional) You can even put the entire bazel build tree in memory if you want to speed up
build times. If you want to do this:

```sh
mv ~/.cache ~/.old-cache
mkdir "/dev/shm/${USER}_cache"
ln -s "/dev/shm/${USER}_cache" ~/.cache
```
You may need to change it back `mv ~/.old-cache ~/.cache` afterward.

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
### Training of Normalized Value Prediction (NVP), Behavior Cloning (BC), and Q value
Run the scripts under the `scripts` folder to start training models of NVP, BC or Q value. The model checkpoints, training log, and configurations will be saved under `./outputs`. The configurations are saved in a file named `args.pkl` and can be used for testing later.


### Testing of Normalized Value Prediction (NVP), Behavior Cloning (BC), and Q value
First we create a directory for gathering testing results: `mkdir cg_paper_exp`. 
Set the number of CPUs/GPUs to use for testing by setting environment variables `NUM_CPU` and `NUM_GPU`. For example, if you want to use 10 CPUs and 1 GPU, you can run `export NUM_CPU=10; export NUM_GPU=1`. 
Run `python rlcompopt/eval_local.py --args_path /path/to/output/args.pkl` to obtain model performance on the validation set and test set. 
There is a [script](scripts/test.sh) for testing all models in the outputs folder. You can modify it and run it `bash scripts/test.sh`.


### Training and testing RL-PPO agents
Run `bash scripts/generate_graph_reward_history_online.sh` to start a group of processes (generators) that do the exploration and send trajectories data to the model for training. 

And at the same time, in another shell, run `bash scripts/train_graph_reward_history_online.sh` to start the trainer of RL-PPO, which receives trajectories data from the generators.

Alternatively, you can run `python scripts/submit_online_train_ppo_action_histogram.py` and `python scripts/submit_ppo_autophase_action_histogram.py` to run all the RL-PPO experiments. You should check the files and provide necessary arguments to the two scripts.

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