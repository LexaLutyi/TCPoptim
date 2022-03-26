using LinearAlgebra
using DifferentialEquations

function queue_output(cin, queue, cout_max)
    if all(queue .<= 0.)
        min.(cin / sum(cin) * cout_max, cin)
    else
        normalize(queue, 1) .* cout_max
    end
end


function d_queue(cin, queue, cout_max, queue_max)
    cout = queue_output(cin, queue, cout_max)
    over = cin - cout
    lost = sum(queue) > queue_max ? over : 0 * over
    dq = over - lost
    dq, cout, lost
end


queue_space(q, Q) = Q - sum(q) .+ q


function f(x, p, t)
    q1, q2 = x[1:2], x[3:4]
    cout0 = [100., 1000.]
    dq1, cout1, lost1 = d_queue(cout0, q1, 110., 1000.)

    queue_max2 = queue_space(q2, 1000.)
    dq21, cout21, lost21 = d_queue(cout1[1], q2[1], 100., queue_max2[1])
    dq22, cout22, lost22 = d_queue(cout1[1], q2[1], 10., queue_max2[1])

    [dq1; dq21; dq22]
end

u0 = zeros(4)

t0 = 0.
t1 = 100.
dt = 0.1

prob = ODEProblem(f, u0, (t0, t1))
sol = solve(prob, tstops=t0:dt:t1)

using Plots

plot(sol)
