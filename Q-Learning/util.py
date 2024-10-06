import numpy as np
import matplotlib.pyplot as plt
import datetime

def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label = '1')
    ax2 = fig.add_subplot(111, label = '2', frame_on = False)
    
    color1 = 'C0'
    ax.plot(x, epsilons, color = color1)
    ax.set_xlabel('Training Steps', color = color1)
    ax.set_ylabel('Epsilon', color= color1)
    ax.tick_params(axis='x', color= color1)
    ax.tick_params(axis='y', color= color1)

    N = len(scores)

    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    color2 = 'C1'
    ax2.scatter(x, running_avg, color=color2)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color=color2)
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', color=color2)

    name, suf = filename.split('.')
    filename = '.'.join([name + '_' + datetime.datetime.now().strftime('%Y%m%d-%H%M'),suf])

    plt.savefig(filename)

 
