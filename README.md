# Automatic Speech Recognition System using spiking neural networks and Hidden Markov Model

The project aimed to model the auditory system using spiking neurons network (with the Nengo platform and the SpiNNaker chip). The spikes provided by the auditory system are analysed using a automatic speech recognition system which recognise five simple speech commands: go, stop, back, right and left. These commands are used to move a robot. The project was divided in three main parts. The first part was to model the human auditory system and map sound frequencies into spikes using the Nengo platform. The second one was to use these signals and recognise a few commands to move a robot with speech. Finally, the third part is a desktop application for  recording the audio and sending it to the neural network to recognise the commands.

## Installation

PyQt4 for python 2.7
https://www.riverbankcomputing.com/software/pyqt/download
run the .exe and install

PyAudio (listen to mic on windows)
pip install pyaudio

Pyqtgraph (work alongside pyqt4)
pip install pyqtgraph

Nengo
git clone https://github.com/nengo/nengo.git
cd nengo
python setup.py develop

SpiNNaker
git clone https://github.com/ctn-waterloo/nengo_spinnaker
cd nengo_spinnaker
python setup.py develop
nengo_spinnaker_setup
1 - Manually
2 - IP: spinn-9.cs.man.ac.uk

Robot:
clone https://github.com/tcstewar/nstbot.git
cd nstbot
python setup.py install


## Usage

To run the application run the audioApplication.py
To train the model run trainModel.py
To record more train samples run the recordAudio.py, then cut the sound as the samples in audio_samples and place it in the correct folder inside audio_samples folder.

## Credits

Created by Leonardo Claudio de Paula e Silva

## License

MIT License

Copyright (c) 2016 Leonardo Claudio de Paula e Silva

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.