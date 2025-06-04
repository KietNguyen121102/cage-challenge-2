import inspect
from pprint import pprint
from CybORG import CybORG
from CybORG.Agents import *
from CybORG.Shared.Actions import *
from statistics import mean, stdev

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'


env = CybORG(path,'sim',agents={'Red':B_lineAgent})

agent = BlueReactRestoreAgent()

results = env.reset('Blue')
obs = results.observation
action_space = results.action_space

file_name = "test_result.txt"
r = []
a = []

for i in range(100):
    action = agent.get_action(obs,action_space)
    results = env.step(action=action,agent='Blue')
    r.append(results.reward)
    obs = results.observation
    a.append((str(env.get_last_action('Blue')), str(env.get_last_action('Red'))))

print(f'Average reward for blue agent is: {mean(r)} with a standard deviation of {stdev(r)}')
with open(file_name, 'a+') as data:
    data.write(f'steps: {i}, mean: {mean(r)}, standard deviation {stdev(r)}\n')
    for act, sum_rew in zip(a, r):
        data.write(f'actions: {act}, total reward: {sum_rew}\n')
   