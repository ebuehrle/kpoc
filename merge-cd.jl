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
d = 4
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
N = 5

m = GMPModel(Mosek.Optimizer)
@variable m ρ Meas(x,support=@set(x'x<=10))
@objective m Min Mom(Λ,ρ)
@constraint m F.U[:,1:N]'*Mom.(differentiate(ϕ,x[1:2])*x[3:4],ρ) .== F.U[:,1:N]'*(integrate.(ϕ,ρT) - integrate.(ϕ,ρ0))
@constraint m F.U[:,N+1:end]'*Mom.(ϕ,ρ) .== 0
optimize!(m)

q = let v = monomials(x[1:2],0:d)
    Σ = integrate.(v*v',ρ)
    v'*inv(Σ+1e-4I)*v
end
save("merge-cd.pdf", Plots.Image((x...)->1/q(x),(0,1),(0,1)))
