import gym

XS = 1
OS = -1
EM = 0

t3 = gym.make('gym_envs:t3-v0').unwrapped

b = t3._board

# test capture move check
b.set(0, 0, XS)
b.set(0, 1, XS)
b.set(0, 2, XS)
print(b)
print(b.winner())
b.reset()

b.set(0, 0, XS)
b.set(1, 0, XS)
b.set(2, 0, XS)
print(b)
print(b.winner())
b.reset()


b.set(0, 2, XS)
b.set(1, 2, XS)
b.set(2, 2, XS)
print(b)
print(b.winner())
b.reset()

b.set(0, 2, XS)
b.set(1, 1, XS)
b.set(2, 0, XS)
print(b)
print(b.winner())
b.reset()


b.set(0, 1, OS)
b.set(1, 1, OS)
b.set(2, 1, OS)
print(b)
print(b.winner())
b.reset()

b.set(1, 0, OS)
b.set(1, 1, OS)
b.set(1, 2, OS)
print(b)
print(b.winner())
b.reset()

b.set(2, 0, OS)
b.set(2, 1, OS)
b.set(2, 2, OS)
print(b)
print(b.winner())
b.reset()

b.set(0, 0, OS)
b.set(1, 1, OS)
b.set(2, 2, OS)
print(b)
print(b.winner())
b.reset()

b.set(0, 0, XS)
b.set(0, 1, XS)
b.set(1, 0, OS)
b.set(1, 1, OS)
print(t3.E(None, (0, 2)))
print(b)
b.reset()

b.place(0, 0, XS)
b.place(1, 0, OS)
b.place(0, 1, XS)
b.place(1, 1, OS)
b.place(2, 2, XS)
print(t3.E(None, (1, 2)))
print(b)
b.reset()

b.set(1, 1, XS)
b.set(2, 1, OS)
b.set(2, 2, XS)
b.set(1, 2, XS)
b.set(0, 0, OS)
print(b)
print(b.winner())


b.set(0, 2, OS)
b.set(0, 1, XS)
b.set(1, 0, OS)
print(t3.E(t3.x(), (2, 0)))
print(b)
print(t3.x())
print(b.winner())
