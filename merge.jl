using CSV, DataFrames
using MomentOpt, MosekTools
using DynamicPolynomials
using PGFPlots

D = CSV.read("vehicle_tracks_000.csv", DataFrame) |>
    (d -> d[:,["x","y","vx","vy"]]) |>
    (d -> d .- [1000 1000 0 0]) |>
    (d -> d ./ 20) |>
    (d -> filter(e -> 0 <= e["x"] <= 1, d)) |>
    (d -> filter(e -> 0 <= e["y"] <= 1, d))

p = Plots.Scatter(D[:,["x","y"]])
save("merge.pdf",p)
