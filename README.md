# urutau-cli
This repository contains the cli for the functionalities of the Urutau® project


### Installation
To use this program, python version 3.6 or later is necessary.
The use of virtualenv is recommended!

```bash
git clone http://github.com/guifonte/urutau-cli
cd urutau-cli/
pip install -r requirements.txt
```

It is also necessary to install the dependencies from pydub for mp3 file conversion.
For a better explanation, go to [pydub's repository](https://github.com/jiaaro/pydub).

### Usage
For usage info type in the terminal:
```bash
python urutau.py --help
```
-OR-
```bash
python urutau.py -h
```
you will get the following list of running options:
```bash
usage: urutau.py [OPTIONS] COMMAND [ARG...]
       urutau.py [ --help | --version ]

Urutau Project - Hybrid-Modal Synthesizer

optional arguments:
  -h, --help  show this help message and exit

Commands:
    plot      General plotter for .wav files
    run       Hybrid Modal Synthesis Method Runner
```
For more information about the commands, run in the terminal:
```bash
python urutau.py COMMAND -h 
```

### To be implemented
* fft comparisons
* list strings
* spectrogram
* fix the implementation with peg coupling

---

Urutau® Project - Turbonius®
