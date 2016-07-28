import nengo
import processes
import nstbot


def run():

    neurons_per_freq = 20  # You can increase this, but low numbers work pretty well
    Fs = 8000
    frame_size = int(Fs * 0.040)    # 40ms
    freqs = frame_size / 2
    duration = 10000

    asr_input = processes.ASRInput()

    # Robot initialization
    # bot = nstbot.PushBot()
    # bot.connect(nstbot.Socket('10.162.177.57'))

    with nengo.Network() as model:
        # Auditory filters
        model.filterbank = processes.FFTProcess(
            sample_rate=Fs, frame_size=frame_size)

        # Inner hair cell activity
        model.ihc = nengo.Node(output=model.filterbank, size_out=freqs)

        # Cochlear neurons projecting down auditory nerve
        model.auditory = nengo.Ensemble(n_neurons=freqs * neurons_per_freq, dimensions=freqs)
        nengo.Connection(model.ihc, model.auditory)

        model.command = nengo.Node(asr_input.step, size_in=freqs, size_out=1)
        nengo.Connection(model.auditory, model.command)


        # Robot
        # def command_input(t, x):
        #     x = round(x)
        #     if x == 0.:
        #         return [0.4, 0.4]
        #     elif x == 1.:
        #         return [-0.4, -0.4]
        #     elif x == 2.:
        #         return [0.4, -0.4]
        #     elif x == 3:
        #         return [-0.4, 0.4]
        #     else:
        #         return [0., 0.]
        #
        # motors = nengo.Node(command_input, size_in=1, size_out=2)
        # nengo.Connection(model.command, motors)
        #
        # def bot_control(t, x):
        #     bot.motor(x[0], x[1])
        #
        # bot_c = nengo.Node(bot_control, size_in=2)
        # nengo.Connection(motors, bot_c)

    sim = nengo.Simulator(model)
    sim.run(duration, progress_bar=False)

