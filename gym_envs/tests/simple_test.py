
import gym

XS = 1
OS = -1
EM = 0

t3 = gym.make('gym_envs:simp-v0').unwrapped

b = t3._board

b.place(0, 0, XS)
b.place(0, 1, XS)
b.place(1, 0, XS)
print(t3.step((0,0)))
print(t3.step((1,1)))
b.reset()

for i in range(2):
    for j in range(2):
        print(t3.step((i, j)))
