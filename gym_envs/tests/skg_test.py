import gym

XS = 1
OS = -1
EM = 0
DS = 0.5

skg = gym.make('gym_envs:skg-v0').unwrapped

b = skg._board

# test capture move check
b.set(0, 0, XS)
b.set(0, 1, OS)
b.set(1, 0, OS)
b.set(8, 0, XS)
b.set(8, 1, OS)
b.set(7, 0, OS)
b.set(0, 8, XS)
b.set(1, 8, OS)
b.set(0, 7, OS)
b.set(8, 8, XS)
b.set(8, 7, OS)
b.set(7, 8, OS)
b.set(4, 2, XS)
b.set(4, 3, OS)
b.set(2, 4, XS)
b.set(3, 4, OS)
b.set(4, 6, XS)
b.set(4, 5, OS)
b.set(6, 4, XS)
b.set(5, 4, OS)
print(b, b.capturing_moves())
print(b.possible_moves())
print(b.dominance())

b.set(2, 2, XS)
o_caps = b.capturing_moves()
print(b, o_caps)

print(skg.x())


# test capturing
for x, y in o_caps:
    b.place(x, y, OS)

print(b)

b.reset()


# test captures w.r.t board orientation
b.set(1, 1, XS)
b.set(1, 2, OS)
b.set(2, 1, OS)
print(b.capturing_moves())
b.place(1, 3, XS)
print(b)
