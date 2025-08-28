using CSV, DataFrames
using MomentOpt, MosekTools
using DynamicPolynomials
using LinearAlgebra
using PGFPlots

D = CSV.read("vehicle_tracks_000.csv", DataFrame) |>
    (d -> d[:,["x","y","vx","vy"]]) |>
    (d -> d .- [1000 1000 0 0]) |>
    (d -> d ./ 20) |>
    (d -> filter(e -> 0 <= e["x"] <= 1, d)) |>
    (d -> filter(e -> 0 <= e["y"] <= 1, d))

@polyvar x[1:4]
d = 3
ϕ = monomials(x[1:2],0:2d)
ρ0 = DiracMeasure(x,[0.5,0.0,0.0,0.0])
ρT = DiracMeasure(x,[0.0,0.6,0.0,0.0])
M = sum(DiracMeasure(x,collect(s)) for s in eachrow(D)) * (1/size(D,1))
Λ = let v = monomials(x,0:d)
    Σ = integrate.(v*v',M)
    v'*inv(Σ+1e-4I)*v
end

Σ = integrate.(ϕ*ϕ',M)
F = svd(Σ)

NN = 1:length(F.S)
VV = zeros(size(NN))
XX = []
for (i,N) in enumerate(NN)
    m = GMPModel(Mosek.Optimizer)
    @variable m ρ Meas(x,support=@set(x'x<=10))
    @objective m Min Mom(Λ,ρ)
    @constraint m F.U[:,1:N]'*Mom.(differentiate(ϕ,x[1:2])*x[3:4],ρ) .== F.U[:,1:N]'*(integrate.(ϕ,ρT) - integrate.(ϕ,ρ0))
    @constraint m F.U[:,N+1:end]'*Mom.(ϕ,ρ) .== 0
    optimize!(m)
    VV[i] = objective_value(m)
    push!(XX, integrate.(ϕ,[ρ]))
end

XX = stack(XX)
EE = sqrt.(sum((XX .- XX[:,end]).^2,dims=1))[:]
save("error.pdf", Plots.Linear(NN,EE))
