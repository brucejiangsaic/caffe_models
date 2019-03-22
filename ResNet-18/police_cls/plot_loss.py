'''
Created on 2017-6-9
@brief: Show loss
@author: Felix Voo(36904)
'''
#encoding:utf-8
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

def read_losses(log_filename, line_pattern):
	fp = open(log_filename)
	pattern = re.compile(line_pattern)
	iters = []
	losses = []
	for line in fp.readlines():
		res = pattern.match(line)
		if res:
			groups = res.groups()
			print(groups)
			iters.append(int(groups[0]))
			losses.append(float(groups[1]))
	fp.close()
	return iters, losses

# 10968: 14.180676, 13.682640 avg, 0.002000 rate, 1.735627 seconds, 701952 images
def show_darknet_loss(log_filename, skip_iters=0):
	line_pattern = '^(\d+): ([0-9\.]+),.*images'
	iters, loss = read_losses(log_filename, line_pattern)
	plt.figure()
	plt.plot(iters[skip_iters:], loss[skip_iters:], '-b')
	plt.grid()
	last_loss = np.asarray(loss[-40:])
	plt.title('Training loss, last loss = %.2f'%np.mean(last_loss))
	plt.show(block=True)

# Iteration 1600 (0.992005 iter/s, 100.806s/100 iters), loss = 8.92929
def show_caffe_loss(log_filename, skip_iters=0):
	line_pattern = '.* Iteration (\d+).*, loss = ([\d\.]+)'
	iters, loss = read_losses(log_filename, line_pattern)
	plt.figure()
	plt.plot(iters[skip_iters:], loss[skip_iters:], '-b')
	plt.grid()
	plt.title('Training loss, last loss = %.2f'%loss[-1])
	plt.show(block=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--src', default = r'1214.log', help = 'Log file')
	parser.add_argument('--skip', default = 10, type=int,
	help = 'Number of skipped iterations')
	parser.add_argument('--arch', default = 'caffe',
	help = 'Training architecture, darknet or caffe')
	args = parser.parse_args()

	if args.arch == 'darknet':
		show_darknet_loss(args.src, args.skip)
	else:
		show_caffe_loss(args.src)
	print('Done.')
