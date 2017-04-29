# Imports.
import numpy as np
import numpy.random as npr
import pandas as pd
import os.path
import datetime
from itertools import product

from SwingyMonkey import SwingyMonkey

from collections import Counter


class Learner(object):
	'''
	This agent jumps randomly.
	'''

	def __init__(self, eta = 0.2, gamma = 1, epsilon = 0.2, mx = 50, my = 25, ms = 1):
		self.n_iters = 0
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		# Store the last velocity independent of the state
		self.last_vel = None

		# Multipliers for horizontal and vertical distance + gravity
		self.mx = mx
		self.my = my
		self.ms = ms

		# This is the cutoff for high/low gravity
		self.high_g_thresh = 2

		# Initialize Q
		# It has keys of the form (state, action)
		self.Q = Counter()

		# Learning rate
		self.eta = eta

		# Discount factor
		self.gamma = gamma

		# Exploration rate
		self.epsilon = epsilon

	def reset(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.last_vel = None

	def action_callback(self, state):
		'''
		Implement this function to learn things and take actions.
		Return 0 if you don't want to jump and 1 if you do.
		'''

		# You might do some learning here based on the current state and the last state.

		# You'll need to select and action and return it.
		# Return 0 to swing and 1 to jump.

		new_state  = self.transform_state(state)
		new_action = self.choose_action(new_state)

		self.last_action = new_action
		self.last_state  = new_state

		return self.last_action

	def reward_callback(self, reward):
		'''This gets called so you can see what reward you get.'''

		### Update Q
		# Get current state and transform it
		try:
			current_state = self.transform_state(self.swing.get_state(), update_vel=False)
		except:
			current_state = self.last_state

		# Maximize Q from current state
		Qmax = max(self.Q[(current_state, 0)], self.Q[(current_state, 1)])

		# Last Qsa value
		Qsa = self.Q[(self.last_state, self.last_action)]
		self.Q[(self.last_state, self.last_action)] = Qsa  - self.eta * (Qsa - (reward + self.gamma * Qmax))

		self.last_reward = reward

	def choose_action(self, state, thresh = 0.08):

		if state[1] > 0:
			return 0

		epsilon = 0.5 * np.exp(-self.n_iters / float(500))

		self.n_iters += 1
		# With probability epsilon, explore
		if npr.uniform() < epsilon:
			# We don't want to explore the space randomly,
			# since we should not jump much more frequently than
			# we do jump
			return 1 if npr.uniform() < thresh else 0
		# Otherwise, follow the optimal policy based on Q
		else:
			Qs1 = self.Q[(state, 1)]
			Qs0 = self.Q[(state, 0)]
			if Qs0 == Qs1:
				return 1 if npr.uniform() < thresh else 0
			else:
				return 1 if Qs1 > Qs0 else 0

	def transform_state(self, state, update_vel = True):
		# Rescaled horizontal distance to next tree 
		dx = state["tree"]["dist"] / self.mx

		# Vertical distance from bottom of monkey to bottom of tree
		dy = (state["monkey"]["top"] - state["tree"]["top"]) / self.my

		# Velocity of the monkey
		vel = state["monkey"]["vel"] / self.ms

		# Determine if there is high or low gravity
		# For the first time step, randomly choose high or low
		if self.last_vel is None:
			gravity = npr.choice([0, 1])
		elif np.abs(state["monkey"]["vel"] - self.last_vel) > self.high_g_thresh:
			gravity = 1

		else:
			gravity = 0


		if update_vel:
			self.last_vel = state["monkey"]["vel"]

		return (dx, dy, vel, gravity)



def run_games(learner, iters = 100, t_len = 100):
	'''
	Driver function to simulate learning by having the agent play a sequence of games.
	'''
	# intialize df 
	df = pd.DataFrame(columns = ["gravity", "score", "death"]) 
	
	# run iters games
	for ii in range(iters):
		# Make a new monkey object.
		swing = SwingyMonkey(sound=False,                  # Don't play sounds.
							 text="Epoch %d" % (ii),       # Display the epoch on screen.
							 tick_length = t_len,          # Make game ticks super fast.
							 action_callback=learner.action_callback,
							 reward_callback=learner.reward_callback)
		learner.swing = swing

		# Loop until you hit something.
		while swing.game_loop():
			pass
		
		# Save score history.
		df.loc[len(df)] = [swing.gravity, swing.score, swing.death]

		# Reset the state of the learner.
		learner.reset()


	return df

def stats(df):
	"""Helper function to get stats from df"""
	vals = [df.score.mean(), df.score.quantile(0.5), 
	        df.score.quantile(0.8), df.score.max(), 
	        df.death.mode()[0]]
	return vals

if __name__ == '__main__':
	etas = [0.15, 0.1, 0.05]
	gammas = [1] 
	epsilons = [0.1, 0.05, 0.01]
	mds = [122, 120, 118]
	mys = [37, 36, 35, 34, 33]
	mss = [54, 53, 52, 51, 50]
	param_list = [etas, gammas, epsilons, mds, mys, mss]
	params = product(*param_list)

	now = datetime.datetime.now()
	print "Starting time: {}".format(str(now))

	i = 0
	for eta, gamma, epsilon, md, my, ms in params: 
		
		### check that test hasn't been run
		# initialize name 
		params = [eta, gamma, epsilon, md, my, ms]
		name = "_".join(map(str, params))

		# initialize logfile number
		if os.path.isfile('csvs_vel_35_52_120/' + name  + ".csv"):
			continue

		### run games
		# Select agent.
		agent = Learner(eta, gamma, epsilon, md, my, ms)

		# Run games, account for bug in distro code
		while True:
			try:
				df = run_games(agent, 100, 0)
			except UnboundLocalError:
				continue
			break

		### log results
		# log all individual scores in csv in folder 
		df.to_csv('csvs_vel_35_52_120/' + name + ".csv", index=0)
	
		# get all summary stats
		full_stats = stats(df)
		_30_above = stats(df[30:])
		_30_above_high = stats(df[30:][df[30:].gravity == 1])
		_30_above_low = stats(df[30:][df[30:].gravity == 4])
		combined = params + full_stats + _30_above + _30_above_high + _30_above_low
		
		# append summary stats to grid_csv
		with open("grid_results_vel_35_52_120.csv", "a") as myfile:
			myfile.write(','.join(map(str,combined)) + "\n")

		# shout at command line
		i += 1
		if i % 25 == 0:
			elapsed = datetime.datetime.now() - now
			print "Combos : {},  Elapsed time: {}".format(i, str(elapsed))
