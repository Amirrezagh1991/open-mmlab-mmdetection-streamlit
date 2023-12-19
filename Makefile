# Makefile for installing Python packages from source

# Conda environment name
#CONDA_ENV_NAME = open-mmlab-mmdetection
#SHELL = zsh

.PHONY: all install clean

all: install download

install:
	# Install packages in requirements.txt
	pip install -r requirements.txt
	pip install -U openmim
	conda install pytorch torchvision cpuonly -c pytorch -y

	# Install mmdetection from source
	git clone https://github.com/open-mmlab/mmdetection.git
	cd mmdetection && pip install -r requirements.txt && pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI" && pip install -v -e .

    # Install mmengine from source
	git clone https://github.com/open-mmlab/mmengine.git
	cd mmengine && pip install -e . -v

    # Install mmcv from source
	git clone https://github.com/open-mmlab/mmcv.git
	cd mmcv && pip install -r requirements/optional.txt && MMCV_WITH_OPS=1 pip install -e .

#download:
    # Create a directory for checkpoints (if it doesn't exist)
	#mkdir -p ./checkpoints

	# Download a pre-trained model using openmim
	#mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest ./checkpoints

clean:
	# Deactivate the Conda environment and remove it
	rm -rf mmdetection
	rm -rf mmcv
	rm -rf mmengine

