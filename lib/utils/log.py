# -------------------------------
# Written by Yongxi Lu
# -------------------------------


""" Utilities to interpret automatic logs generated by the training procedure
"""

import re
import numpy as np

# necessary when used without a display
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def parse_mle_and_plot(filename, sets, output='loss.png', run_length = 1, max_iters = None, epoch_size = None):
    """ parse the mle log for a particular set, output file at the output location"""
    
    # parse from files    
    iters = {}
    err = {}
    for set in sets:
        iters[set], err[set] = parse_mle(filename, set)

    # if max_iters is specified, truncate loss
    if max_iters is not None:
        for set in sets:
            end = min(max_iters, np.max(iter[set]))
            iters[set] = iters[set][:end]
            err[set] = err[set][:end]

    # perform sampling and smoothing for the training set
    if 'training' in sets and 'validation' in sets:
        A = iters['training']
        B = iters['validation']
        # sample iterations
        center_inds = np.array([i for i in xrange(len(A)) if A[i] in B])
        iters['training'] = A[center_inds]
        # sample errors
        begins = np.maximum(0, center_inds - run_length)
        ends = np.minimum(center_inds + run_length, len(A))
        idx = np.vstack((begins, ends)).transpose()
        values = np.array([np.mean(err['training'][idx[i,0]:idx[i,1], :], axis=0) for i in xrange(len(center_inds))])
        err['training'] = values

    # convert iterations to epochs if necessary
    if epoch_size is not None:
        for set in sets:
            iters[set] = iters[set].astype(np.float32, copy=False)
            iters[set] /= epoch_size

    plot_mle_mean(iters, err, output)

def parse_mle(filename, set):
    """ parse mle log for a particular set """
    
    iters = []
    err = []
    pattern = 'Iteration [0-9]*: {} error = \[[0-9.\ ]*\]'.format(set)
    with open(filename) as f:
        data = ' '.join([line.replace('\n', '') for line in f])
        match = re.findall(pattern, data)
        for i in xrange(len(match)):
            parts = match[i].split(' = ')
            # match iterations
            iters.append(int(re.search('[0-9]+',parts[0]).group()))
            # match error rates
            err_str = re.search('[0-9.\ ]+', parts[1]).group().split()
            err.append([float(err_str[i]) for i in xrange(len(err_str))])
                    
    iters = np.array(iters)
    err = np.array(err)

    return iters, err

def plot_mle_mean(iters, err, output="loss.png"):
    """ Plot the evolution of multi-label error as a function of training iterations """
   
    fig, ax = plt.subplots()
    ax.set_autoscale_on(False)
    max_value = 0
    max_iters = 0

    # set y axis [0, max], set x axis [0, max_iters] (if the provided max_iters is not integer, 
    # e.g. it is the number of epochs, round it)
    for set, x in iters.iteritems():
        ax.plot(x, np.mean(err[set], axis=1), label=set, linewidth=3.0)
        max_value = max(max_value, np.max(np.mean(err[set], axis=1)))
        max_iters = np.round(max(max_iters, np.max(iters[set]))).astype(np.int32)

    ax.axis([0, max_iters, 0, max_value])
    legend = ax.legend(loc='best', shadow=True)

    plt.xlabel('Number of iterations')
    plt.ylabel('Error rate')
    plt.title('Multi-label error as function of training iterations')
    # save the figure
    plt.savefig(output)

if __name__ == '__main__':

    iters_train, err_train = parse_mle('test_parse.txt', 'training')
    iters_val, err_val = parse_mle('test_parse.txt', 'validation')

    print 'Parsing results for training'
    print iters_train
    print err_train
    
    print 'Parsing results for testing'
    print iters_val
    print err_val 

    iters = {'training': iters_train, 'validation': iters_val}
    err = {'training': err_train, 'validation': err_val}

    plot_mle_mean(iters, err)
