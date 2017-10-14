import numpy as np
import random
from collections import defaultdict
import motion_model_reward as mmr


class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table =  defaultdict(lambda: [0.0,0.0,0.0,0.0,0.0])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)



if __name__ == "__main__":
    agent = QLearningAgent(actions = list(range(5)))

    for episode in range(1000):
        state = [mmr.x1_in,mmr.y1_in]
        timestep = 0
        while True:

            action = agent.get_action(str(state))
            next_state,reward ,done,timestep = mmr.step(action,timestep)
            agent.learn(str(state), action, reward, str(next_state)) #reward defined by the sum of heuristics

            state = next_state

            if done:
                break
        mmr.plots()
