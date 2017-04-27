# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

from collections import Counter


class Learner(object):
	'''
	This agent jumps randomly.
	'''

	def __init__(self, eta = 0.2, gamma = 1, epsilon = 0.2):
		self.n_iters = 0
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		# Store the last velocity independent of the state
		self.last_vel = None

		# Multipliers for horizontal and vertical distance
		self.md = 50
		self.my = 25

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
		self.epsilon = 0.2

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
		dx = state["tree"]["dist"] / self.md

		# Vertical distance from bottom of monkey to bottom of tree
		dy = (state["monkey"]["top"] - state["tree"]["top"]) / self.my

		# Velocity of the monkey
		vel = state["monkey"]["vel"]

		# Determine if there is high or low gravity
		# For the first time step, randomly choose high or low
		if self.last_vel is None:
			gravity = npr.choice([0, 1])
		elif np.abs(vel - self.last_vel) > self.high_g_thresh:
			gravity = 1
		else:
			gravity = 0

		if update_vel:
			self.last_vel = vel

		return (dx, dy, vel > 0, gravity)




def run_games(learner, hist, iters = 100, t_len = 100):
	'''
	Driver function to simulate learning by having the agent play a sequence of games.
	'''
	
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
		hist.append(swing.score)

		# Reset the state of the learner.
		learner.reset()
		
	return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 100, 1)

	# Save history. 
	np.save('hist',np.array(hist))


