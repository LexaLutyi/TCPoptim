using Plots
using DifferentialEquations

function priority_output(v, C)
    min.(v / sum(v) * C, v)
end


function queue_space(q, Q)
    Q - sum(q) .+ q
end


function p_loss(q, Q)
    a = q / Q
    w = 0.5
    min((a > w) * (a - w) / (1 - w), 1.)
end

plot(0.:0.1:queue_max, x -> p_loss(x, queue_max))


function diff_simple_queue(queue, input, max_output, queue_max)
    output = queue > 0 ? max_output : min(input, max_output)
    A = input - output
    lost = p_loss(queue, queue_max) * (A > 0) * A
    dq = A - lost
    [dq, output, lost]
end


function restore(small, ixs)
    n = sum(length.(ixs))
    z = zeros(n)
    for (i, ix) in enumerate(ixs)
        z[ix] .= small[i]
    end
    z
end


function diff_queue(queues, inputs, ix_out, C_out, queue_max)
    Q = queue_space(queues, queue_max)

    C = restore(C_out, ix_out)
    pre_out = mapreduce(diff_simple_queue, hcat, queues, inputs, C, Q)[2, :]
    pre_out = getindex.(Ref(pre_out), ix_out)
    output = reduce(vcat, priority_output.(pre_out, C_out))

    # dq
    # output
    # lost
    mapreduce(diff_simple_queue, hcat, queues, inputs, output, Q)
end


function loss_detection(u, t, int)
    lost = pref(u, int.p, t).lost
    sum(abs, lost) > 0
end


function loss_affect!(int)
    lost = pref(int.u, int.p, int.t).lost
    ix = abs.(lost) .> 0.
    @views w = int.u[1:3]
    w[ix] ./= 2
end


loss_cb = DiscreteCallback(loss_detection, loss_affect!)


function pref(u, p, t)
    ix_out, C_out, queue_max, τ = p
    w, q = u[1:3], u[4:6]
    RTT = sum.(getindex.(Ref(q), ix_out)) ./ C_out .+ τ

    input = w ./ restore(RTT, ix_out)
    dq, output, lost = diff_queue(q, input, ix_out, C_out, queue_max) |> eachrow

    (; w, q, RTT, input, output, lost, dq)
end


function f(du, u, p, t)
    dq = pref(u, p, t).dq
    dw = ones(3) ./ u[1:3]
    du .= [dw; dq]
end


n = 3
q = rand(n)
w = 10 * rand(n)
u0 = [w; q]
du0 = zeros(2n)


ix_out = [1:2, 3:3]
C_out = [1., 2.]
queue_max = sum(C_out) * 5
τ = 1.
p = (ix_out, C_out, queue_max, τ)

t0 = 0.
t1 = 200.
dt = 0.01

prob = ODEProblem(f, u0, (t0, t1), p)
sol = solve(prob, Euler(), callback=loss_cb, tstops=t0:dt:t1)

d = mapreduce(hcat, tuples(sol)) do (u, t)
    collect(pref(u, p, t))
end

get_value(d, i) = reduce(hcat, d[i, :])'

w = get_value(d, 1)
q = get_value(d, 2)
RTT = get_value(d, 3)
input = get_value(d, 4)
output = get_value(d, 5)
lost = get_value(d, 6)
dq = get_value(d, 7)

ts = sol.t
plot(ts, w, title="window")
plot(ts, [q sum(q, dims=2)], title="queue")
plot(ts, RTT, title="round trip time")
plot(ts, input, title="input")
plot(ts, output, title="output")
plot(ts, lost, title="lost")
plot(ts, dq, title="δqueue")
