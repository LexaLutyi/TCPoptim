using DifferentialEquations
using Plots


function p(q, Q)
    a = q / Q
    w = 0.9
    min((a > w) * (a - w) / (1 - w), 1.)
end

plot(0.:0.1:3., x -> p(x, 2))

function Q1(q, input, C, Qmax)
    output = q > 0 ? C : min(input, C)
    A = input - output
    lost = p(q, Qmax) * (A > 0) * A
    dq = A - lost
    [dq, output, lost]
end


input(w, q, C, τ) = w / (q / C + τ)


reduce(hcat, Q1.([1., 2.], 1, 1., 1))


function f(dx, x, p, t)
    C, Qmax, τ = p
    w = x[1:2]
    q = x[3:4]

    Q = Qmax .- (sum(q) .- q)
    
    v = Q1.(q, input.(w, q, C, τ), C, Q)
    dq, output, lost = eachrow(reduce(hcat, v))
    dw = [1, w[2]]

    dx .= [dw; dq]
end


function condition(u, t, int)
    q = u[3:4]
    p(sum(q), int.p[2]) > 0
end

function affect!(int)
    int.u[2:2] ./= 2
end

loss_event = DiscreteCallback(condition, affect!)


C = 100.

Qmax = C
τ = 1

x0 = [C, C + 20., 0., 0.]
t0 = 0.
t1 = 200.
dt = 0.01
prob = ODEProblem(f, x0, (t0, t1), (C, Qmax, τ))

sol = solve(prob, Euler(), 
    abstol=1e-8, 
    reltol=1e-8, 
    tstops=t0:dt:t1,
    callback=loss_event
    )


ext_sol = map(tuples(sol)) do (u, t)
    w = u[1:2]
    q = u[3:4]

    u1 = input.(w, q, C, τ)
    Q = Qmax .- (sum(q) .- q)
    dq, out, lost = collect.(zip(Q1.(q, u1, C, Q)...))
    q, dq, out, lost, u1, w
end

q = mapreduce(s -> s[1], hcat, ext_sol)'
dq = mapreduce(s -> s[2], hcat, ext_sol)'
output = mapreduce(s -> s[3], hcat, ext_sol)'
lost = mapreduce(s -> s[4], hcat, ext_sol)'
u1 = mapreduce(s -> s[5], hcat, ext_sol)'
win = mapreduce(s -> s[6], hcat, ext_sol)'

plt_q = plot(sol.t, q, label="q")
plt_io = plot(sol.t, [u1[:, 1], output[:, 1], win[:, 1]], label=["input" "output" "window"])
plt_io2 = plot(sol.t, [u1[:, 2], output[:, 2], win[:, 2]], label=["input" "output" "window"])
plt_dq = plot(sol.t, [dq[:, 1], lost[:, 1]]; label=["dq" "lost"])
plt_dq2 = plot(sol.t, [dq[:, 2], lost[:, 2]]; label=["dq" "lost"])
plot(plt_q, plt_io, plt_io2, plt_dq, plt_dq2, layout=(5, 1))
