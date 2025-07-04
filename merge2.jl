using CSV, DataFrames
using MomentOpt, MosekTools
using DynamicPolynomials
using LinearAlgebra
using PGFPlots

D = CSV.read("vehicle_tracks_000.csv", DataFrame) |>
    (d -> d[:,["frame_id","x","y","vx","vy"]]) |>
    (d -> d .- [0 1000 1000 0 0]) |>
    (d -> d ./ [1 20 20 20 20]) |>
    (d -> filter(e -> 0 <= e["x"] <= 1, d)) |>
    (d -> filter(e -> 0 <= e["y"] <= 1, d))

@polyvar x[1:4]
d = 3
ϕ = monomials(x[1:2],0:2d)
ρ0 = DiracMeasure(x,[0.5,0.0,0.0,0.0])
ρT = DiracMeasure(x,[0.2,0.6,0.0,0.0])
μ0 = DiracMeasure(x,[0.8,0.2,0.0,0.0])
M = sum(DiracMeasure(x,collect(s[2:end])) for s in eachrow(D)) * (1/size(D,1))
Λ = let v = monomials(x,0:d)
    Σ = integrate.(v*v',M)
    v'*inv(Σ+1e-4I)*v
end

frames = unique(D[:,"frame_id"])
M2 = [let Df = filter(e -> e["frame_id"] == f, D); 
    sum(DiracMeasure(x,collect(s[2:end])) for s in eachrow(Df)) 
end for f in frames]
Σ = stack(integrate.(ϕ,m) for m in M2)
F = svd(Σ)
N = 25

m = GMPModel(Mosek.Optimizer)
@variable m ρ Meas(x,support=@set(x'x<=10))
@variable m μ Meas(x,support=@set(x'x<=10))
@variable m μT Meas(x,support=@set(x'x<=10))
@objective m Min Mom(Λ,ρ+μ)
@constraint m F.U[:,1:N]'*Mom.(differentiate(ϕ,x[1:2])*x[3:4],ρ) .== F.U[:,1:N]'*(integrate.(ϕ,ρT) - integrate.(ϕ,ρ0))
@constraint m F.U[:,1:N]'*(Mom.(differentiate(ϕ,x[1:2])*x[3:4],μ) - Mom.(ϕ,μT)) .== F.U[:,1:N]'*(-integrate.(ϕ,μ0))
@constraint m F.U[:,N+1:end]'*Mom.(ϕ,ρ+μ) .== 0
@constraint m Mom(1,ρ) == Mom(1,μ)
optimize!(m)

q = let v = monomials(x[1:2],0:d)
    Σ = integrate.(v*v',ρ) + integrate.(v*v',μ)
    v'*inv(Σ+1e-4I)*v
end
save("merge2.pdf", Axis([
    Plots.Image((x...)->1/q(x),(0,1),(0,1)),
    Plots.Quiver(
        D[1:10:end,"x"],D[1:10:end,"y"],
        D[1:10:end,"vx"]/10,D[1:10:end,"vy"]/10,
        style="-stealth,blue,no markers"),
    Plots.Scatter(integrate.(x[1:2],[ρ0 ρT μ0 μT])),
],xmin=0,xmax=1,ymin=0,ymax=1))
