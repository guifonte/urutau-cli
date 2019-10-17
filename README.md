# urutau-cli
This repository contains the cli for the functionalities of the Urutau® project


### Installation
To use this program, python version 3.6 or later is necessary.
The use of virtualenv is recommended!

```bash
git clone http://github.com/guifonte/urutau-cli
pip install -r requirements.txt
```

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
usage: methodrunner.py [-h] --phikyz <path> [--phikyzp <path>] --fmk <path>
                       [--nb <num>] [--xsi-fixed <val> | --xsi-path <path>]
                       --string <name> [--string-len <len>]
                       [--string-fmax <len>] [--string-f0 <freq>] --xp <pos>
                       [--pluck-ti <time>] [--pluck-dp <len>]
                       [--pluck-F0 <val>] [--pluck-gamma <rad>] --tf <dur>
                       [--method <method>] [--pol <num>] [--damp <method>]
                       [--fs <freq>] [-v] [-g] [--fftwindow <dur>]
                       [--pluckingpoint] [--displ] [--vel] [--acc-no]
                       [--mp3 | --mp3-only]

Hybrid Modal Synthesis Method Runner
------------------------------------
   Required arguments:
   --phikyz <path>
   --fmk <path>
   --string <name>
   --tf <dur>
   --xp <pos>
------------------------------------

optional arguments:
  -h, --help           show this help message and exit

body:
  --phikyz <path>      Path of the PHIKYZ file at the bridge (REQUIRED)
  --phikyzp <path>     Path of the PHIKYZ file at the peg
  --fmk <path>         Path of the FMK file (REQUIRED)
  --nb <num>           body modes (default: max)
  --xsi-fixed <val>    fixed xsi for all modes (default: 0.1)
  --xsi-path <path>    Path of the XSI file

string:
  --string <name>      string name (REQUIRED)
  --string-len <len>   string length (meters) (default: 0.65)
  --string-fmax <len>  max frequency of string modes (Hz) (default: 10000)
  --string-f0 <freq>   fundamental frequency (Hz) (default: string default f0)

pluck:
  ramp function

  --xp <pos>           pluck position (meters)(0 ref: peg)(REQUIRED)
  --pluck-ti <time>    starting time of the ramp (seconds)(default: 0.001)
  --pluck-dp <len>     ramp length (seconds)(default: 0.008)
  --pluck-F0 <val>     height of the ramp (N)(default: 10)
  --pluck-gamma <rad>  pluck angle (radians)(default: pi/2)

simulation:
  --tf <dur>           Duration of the simulation (seconds) (REQUIRED)
  --method <method>    simulation method: 'cfc' or 'fft' (default: cfc)
  --pol <num>          the number of polarizations: 1 or 2 (default: 2)
  --damp <method>      damping method: 'woodhouse' or 'paiva' (default: paiva)
  --fs <freq>          sample frequency (default: 44100)

output:
  -v, --verbose        print progress percentage for cfc
  -g, --graphic        generate pdf with graphics
  --fftwindow <dur>    size of the fft window (ms)(default: full signal)
  --pluckingpoint      generate files for plucking point when cfc
  --displ              generate files for displacement when cfc
  --vel                generate files for velocity when cfc
  --acc-no             does not generate files for acceleration
  --mp3                generate mp3 file
  --mp3-only           generate only mp3 file (no wav)
```

### To be implemented
* fft comparisons
* list strings
* convertion to mp3 files
* spectrogram
* fix the implementation with peg coupling

---

Urutau® Project - Turbonius®
