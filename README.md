# Upscaling network training 

Dataset used (not included):

- [OpenImages v7](https://storage.googleapis.com/openimages/web/index.html) under [Creative Common by 4.0](https://creativecommons.org/licenses/by/4.0/) license
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) - used for research. As provided: <br>
"This dataset is made available for academic research purpose only. All the images are collected from the Internet, and the copyright belongs to the original owners. If any of the images belongs to you and you would like it removed, please kindly inform us, we will remove it from our dataset immediately"

- Flickr2K (NTIRE2017 by SNU_CVLab) — collected using the Flickr API by the [NTIRE2017](https://data.vision.ee.ethz.ch/cvl/ntire17//) [SNU_CVLab](https://cv.snu.ac.kr/) team; copyright remains with original image owners.

This repository contain *only* code used for training models that were created using datasets above and final network weights.

## Training

Code used for training network can be found [here](https://github.com/283018/comp_vision).


## Installation:


*sh/bash/zsh/fish/xonsh/nu/elvish/osh/ngs:*

### Using uv:


```sh
$ uv tool install git+https://github.com/283018/comp_vision_gui.git

# then run gui
comp_vision-gui

# or cli
comp_vision-cli --help
```

### Manual installation:
```sh
git clone https://github.com/283018/comp_vision_gui.git
# or download and unpack release archive

cd comp_vision_gui

# one-time run
uv run comp_vision-gui
# or
uv run comp_vision-cli --help

# or permanent installation:
uv sync

# after .venv created:
uv run comp_vision-gui
# or
uv run comp_vision-cli --help

```

