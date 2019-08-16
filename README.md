# Lungs image segmentation U-net
Repo created for learning and experimenting with Keras AI framework.

## Quick config:  
1. Download Miniconda for Windows from [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).  
2. Install Miniconda  
   - Install for **Just Me**  
   - Add Anaconda to my **PATH enviromantal variables**   
   - **Untick** "*Register Anaconda as my default Python 3.7*" (If you have another python copy installed)     
3. In cmd: 
   - `conda create -n tfgpu tensorflow-gpu SimpleITK pillow keras matplotlib scikit-image pip opencv`  
   - `activate tfgpu`    
4. Run `main.py` using `tfgpu` pyenv with cmd or your favorite IDE (e.g. VS Code, Spyder, Atom) configured to work with python envs.

## Useful cmds:
1. List envs: `conda info --envs`
2. Install package on env: `conda install -n env_name pypng`
3. Add channel to conda: `conda config --env --add channels channel_name`
4. Remove channel from conda: `conda config --remove channels channel_name`
5. Show channels: `conda config --show channels`

## Installing pip modules within miniconda env:
1. Install pip inside your env: `conda install -n env_name pip`
2. Get inside env with Anaconda prompt: `conda activate env_name`
3. Install module using pip inside env: `pip install module_name`