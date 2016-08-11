import nengo
import scipy.io.wavfile as wavfile
import processes
import nengo_spinnaker

import os
for folder in os.listdir("./audio_samples"):
    print('----------'+folder+'-8k-samples.wav'+'--------------')

    # Erase previous pattern
    open(folder+'.bin', 'wb').close()
    (Fs, y) = wavfile.read("./audio_samples/"+folder+'/'+folder+'-8k-samples.wav')

    neurons_per_freq = 20  # You can tweak this, but low numbers work pretty well
    frame_size = int(Fs * 0.040)    # 40ms
    frame_step = int(Fs * 0.010)    # 10ms
    freqs = frame_size / 2
    duration = len(y)/(frame_step * 1000.)

    asr_input = processes.ASRInputTrain(folder+'.bin')

    with nengo.Network() as model:
        # Auditory filters
        model.filterbank = processes.FFTProcessTrain(
            y, sample_rate=Fs, frame_size=frame_size)

        # Inner hair cell activity
        model.ihc = nengo.Node(output=model.filterbank, size_out=freqs)

        # Cochlear neurons projecting down auditory nerve
        model.auditory = nengo.Ensemble(n_neurons=freqs * neurons_per_freq, dimensions=freqs)
        nengo.Connection(model.ihc, model.auditory)

        model.command = nengo.Node(asr_input.step, size_in=freqs, size_out=1)
        nengo.Connection(model.auditory, model.command)


    # Run the model for a second
    sim = nengo.Simulator(model)
    sim.run(duration, progress_bar=True)

    # SpiNNaker
    # sim = nengo_spinnaker.Simulator(model)
    # sim.run(duration)

