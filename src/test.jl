using DifferentialEquations
using Plots


function Q1(q, input; C, Qmax)
    output = q > 0 ? C : min(input, C)
    A = input - output
    lost = (q > Qmax) * (A > 0) * A
    dq = A - lost
    dq, output, lost
end

input(w, q, C, τ) = w / (q / C + τ)

function f(dx, x, p, t)
    w, C, Qmax, τ = p
    q = x[1]

    dq, out, lost = Q1(q, input(w(t), q, C, τ); C, Qmax)
    dx[1] = dq
end


C = 100.
w(t) = C + 10* sin(0.9 * t)
Qmax = 5
τ = 1

x0 = [0.]
t0 = 0.
t1 = 30.
dt = 0.1
prob = ODEProblem(f, x0, (t0, t1), (w, C, Qmax, τ))

sol = solve(prob, abstol=1e-8, reltol=1e-8, tstops=t0:dt:t1)
# plot(sol)


ext_sol = map(tuples(sol)) do (u, t)
    q = u[1]
    u1 = input(w(t), q, C, τ)
    dq, out, lost = Q1(q, u1; C, Qmax)
    q, dq, out, lost, u1
end




q, dq, out, lost, u1 = eachrow(reduce(hcat, collect.(ext_sol)))

plt_q = plot(sol.t, q, label="q")
plt_io = plot(sol.t, [u1, out, w], label=["input" "output" "window"])
plt_dq = plot(sol.t, [dq, lost]; label=["dq" "lost"])
plot(plt_q, plt_io, plt_dq)
