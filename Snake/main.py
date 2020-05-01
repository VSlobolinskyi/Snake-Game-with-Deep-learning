from snake_env import snake_env
from random import randrange
import os
import time

def test():
    env = snake_env()
    for i in range(10):
        _, rew, obs, _ = env.step(randrange(0,4))
        print("Reward for this step is:",rew)
        for i2 in obs:
            for i3 in i2:
                if i3 == 0.1:
                    print(1, end="")
                if i3 == 0.5:
                    print(5, end="")
                if i3 == 0.8:
                    print(7, end="")
                if i3 == 0.9:
                    print(9, end="")
                if i3 == 0.0:
                    print(0, end="")
            print()
        time.sleep(1)
        if i != 9:
            os.system('cls')

test()
        