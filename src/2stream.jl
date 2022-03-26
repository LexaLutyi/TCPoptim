using LinearAlgebra
using DifferentialEquations
using Plots

C0 = 110
C1 = 100
C2 = 10

Q = 1000

maxQ(Q, q...) = Q - sum(q)

upq(q, input, output, maxq) = max(min(q + input - output, maxq), 0)

q1, q2 = 150, 53
min(q1, C0), min(q2, C0)

compout(q1, q2, C) = min(q1, C, max(C - min(q2, C), C / 2))
oneout(q1, C) = min(q1, C)

function f(q, p, t)
    C0, C1, C2, Q = p
    q1, q2, q3, q4, o1, o2, u1, u2 = q
    v1 = compout(q1, q2, C0)
    v2 = compout(q2, q1, C0)
    y1 = oneout(q3, C1)
    y2 = oneout(q4, C2)
    [
        upq(q1, u1, v1, maxQ(Q, q2))
        upq(q2, u2, v2, maxQ(Q, q1))
        upq(q3, v1, y1, maxQ(Q, q4))
        upq(q4, v2, y2, maxQ(Q, q3))
        o1 + y1
        o2 + y2
        0.8u1 + 0.2y1 + 0.1
        0.8u2 + 0.2y2 + 0.1
    ]
end

q = [zeros(6); 100; 10]
p = C0, C1, C2, Q
f(q, p, 0)

n = 1000
prob = DiscreteProblem(f, q, (0, n), p)
sol = solve(prob)
plot(sol)
# v1 = iszero(q1) ? 0 : C / (iszero(q1) + iszero(q2))
# v2 = iszero(q2) ? 0 : C / (iszero(q1) + iszero(q2))
# y1 = iszero(q3) ? 0 : C3
# y2 = iszero(q4) ? 0 : C4
# mq1 = Q - q2
# mq2 = Q - q1
# mq3 = Q - q4
# mq4 = Q - q3
# q1[T + 1] = q1[T] + u1 - v1
# q2[T + 1] = q2[T] + u2 - v2
# q3[T + 1] = q3[T] + v1 - y1
# q4[T + 1] = q4[T] + v2 - y2
