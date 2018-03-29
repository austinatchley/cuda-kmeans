#!/usr/bin/env python3

import sys
import csv
import subprocess
import numpy as np

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from io import StringIO

def plot_bars(barGroups, barNames, groupNames, colors, ylabel='', title='', width=0.8):
    '''Plot a grouped bar chart
    barGroups  - list of groups, where each group is a list of bar heights
    barNames   - list containing the name of each bar within any group
    groupNames - list containing the name of each group
    colors     - list containing the color for each bar within a group
    ylabel     - label for the y-axis
    title      - title
    '''
    fig, ax = plt.subplots()
    offset = lambda items, off: [x + off for x in items]

    maxlen = max(len(group) for group in barGroups)
    xvals = range(len(barGroups))
    
    for i, bars in enumerate(zip(*barGroups)):
        print(bars)
        plt.bar(offset(xvals, i * width/maxlen), bars, width/maxlen, color=colors[i])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(offset(xvals, width / 2))
    ax.set_xticklabels(groupNames)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(barNames, loc='upper left', bbox_to_anchor=(1, 1))


TESTS = 4

if len(sys.argv) < 4:
    print(
            './harness -c [clusters] -t [threshold] -i [iterations] [opional: graph name]'
    )
    sys.exit(0)

c = sys.argv[1]
t = sys.argv[2]
i = sys.argv[3]

if len(sys.argv) >= 5:
    graph_name = sys.argv[4]
else:
    graph_name = 'speedup_graph.pdf'

args = './kmeans.out -c ' + c +    \
    '  -t ' + t +                 \
    ' -i ' + i  +                 \
    ' -I sample/sample.in'


def do_test(i, infile):
    arg_list = args.split()
    arg_list[8] = 'sample/' + infile

    print(' '.join(str(val) for val in arg_list))

    ret_val = 0
    try:
        ret_val = run(arg_list)
    except:
        print('Error. Skipping test case')
        ret_val = -1
    return ret_val


def run(arg_list):
    output = subprocess.check_output(arg_list)
    result = output.decode('utf-8')

    f = StringIO(result)
    reader = csv.reader(f, delimiter=',')
    data = []
    for row in reader:
        data.append(' '.join(element.rstrip() for element in row))

    if i != -1:
        print('Points:\t', data[0])
        print('Iterations:\t', data[1])
        print('Duration:\t', data[2])

    return float(data[2])


def control_test():
    arg_list = args.split()

    output = subprocess.check_output(arg_list)
    result = output.decode('utf-8')

    f = StringIO(result)
    reader = csv.reader(f, delimiter=',')
    data = []
    for row in reader:
        data.append(' '.join(element.rstrip() for element in row))

    print('Duration:', float(data[2]))
    return float(data[2])

files = ['random-n2048-d16-c16.txt', 'random-n16384-d24-c16.txt', 'random-n65536-d32-c16.txt']

controls = {'random-n2048-d16-c16.txt': 0.0269375625, 'random-n16384-d24-c16.txt': 0.2674525, 'random-n65536-d32-c16.txt': 1.49886425}
speedups = []

print('Control')
control_test()
control_test()
print('')
for f in files:
    do_test(-1, f)

    both = 0.0
    tests_completed = 0

    for i in range(TESTS):
        print('\nIteration ', i, 'with file ', f, '.')
        val = do_test(i, f)
        if val != -1:
            both += val
            tests_completed += 1

    average = both / tests_completed
    speedups.append(controls[f] / average)

print('\nSpeedups:')
for s in speedups:
    print(s)

cm = plt.get_cmap('plasma')
color = '#1f10e0'
colors = [cm(i/len(files)) for i in range(len(files))]

plot_bars([speedups], files, ['Files'], colors, 'Speedup', 'Speedup From Sequential CPU Solution With Shared Memory Optimizations')

plt.savefig(graph_name, bbox_inches='tight', format='pdf')
