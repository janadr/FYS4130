using Random
using PyPlot
using Statistics


function periodic(spinIndex, dimension)
	return (spinIndex - 1 + dimension) % dimension + 1
end

function findRandomSpin(dimensions)
	randSpin = zeros(Int64, length(dimensions))
	for i in 1:length(dimensions)
		randSpin[i] = rand(1:dimensions[i])
	end
	return randSpin
end

function findNeighbours(index, lattice, randSpin, dimensions)
	neighbours = vcat([copy(randSpin)], [copy(randSpin)])
	for i in 1:2
		neighbours[i][index] = periodic(neighbours[i][index] + (-1)^i, dimensions[index])
	end
	return neighbours
end

function findAllNeighbours(lattice, randSpin, dimensions)
	nDims = length(dimensions)
	neighbours = []
	for i in 1:nDims
		neighbours = vcat(neighbours, findNeighbours(i, lattice, randSpin, dimensions))
	end
	#println(neighbours)
	return neighbours
end

function calculateLatticeElements(lattice)
	nDims = ndims(lattice)
	elements = 1
	for i in 1:nDims
		elements *= size(lattice, i)
	end
	return elements
end

function constructCluster(randSpin, lattice, dimensions, p)
	nElements = calculateLatticeElements(lattice)
	lattice[randSpin...] *= -1
	cluster = []
	append!(cluster, [randSpin])
	neighbours = findAllNeighbours(lattice, randSpin, dimensions)
	neighbours_copy = copy(neighbours)
	while length(neighbours) > 0
		for i in length(neighbours)
			neighbours_copy = neighbours_copy[1:size(neighbours_copy, 1) .!= i]
			if lattice[randSpin...] == lattice[neighbours[i]...]*(-1) && rand() < p
				append!(cluster, [neighbours[i]])
				lattice[neighbours[i]...] *= -1
				neighbours_copy = vcat(neighbours_copy, findAllNeighbours(lattice, neighbours[i], dimensions))
			end
			if length(cluster) >= nElements
				neighbours_copy = []
				break
			end
		end
		neighbours = copy(neighbours_copy)
	end
	return lattice
end

function calculateCorrelation1D(lattice)
	dim = size(lattice, 1)
	distance = zeros(Int64, dim)
	s = zeros(Int64, dim)
	for i in 1:dim
		distance[i] += lattice[1, 1].*lattice[i, 1]
		s[i] += 1
	end
	c = distance./s
	return c
end


function analyticalCorrelation1D(distance, dimensions, T)
	dim = dimensions[1]
	CR = ((cosh(1/T)/sinh(1/T)).^distance*tanh(1/T)^dim + tanh(1/T).^distance)/(1 + tanh(1/T)^dim)
	return CR
end

function runModel(dimensions, T; nEpochs=10, nCycles_equil=1000, nCycles_meas=5000, J=1)
	p = 1 - exp(-2*J/T)
	c = zeros(dimensions)
	cSamples = []
	mSamples = []
	m2Samples = []
	m4Samples = []
	for epoch in 1:nEpochs
		lattice = rand(-1:2:1, dimensions)
		for cycle in 1:nCycles_equil
			randSpin = findRandomSpin(dimensions)
			lattice = constructCluster(randSpin, lattice, dimensions, p)
		end
		m = mean(lattice)
		m2 = mean(lattice)^2
		m4 = mean(lattice)^4
		for cycle in 1:nCycles_meas
			randSpin = findRandomSpin(dimensions)
			lattice = constructCluster(randSpin, lattice, dimensions, p)
			c .+= calculateCorrelation1D(lattice)
			m += mean(lattice)
			m2 += mean(lattice)^2
			m4 += mean(lattice)^4
		end
		c ./= nCycles_meas
		m /= nCycles_meas
		m2 /= nCycles_meas
		m4 /= nCycles_meas
		append!(cSamples, [c])
		append!(mSamples, m)
		append!(m2Samples, m2)
		append!(m4Samples, m4)
	end
	return mean(cSamples), mean(mSamples), mean(m2Samples), mean(m4Samples)
end



Random.seed!(42)

#=
nEpochs = 10
nCycles_equil = 10000
nCycles_meas = 5000

L = 16
dimensions = L
J = 1

c, m, m2, m4 = runModel(dimensions, 1)

distance = LinRange(0, L-1, L)

fig, ax = subplots(2, 1, figsize=(10, 8), sharex=true, sharey=true)
ax[1].plot(distance, c, color="black", linestyle="dotted", label="Numerical", linewidth=2)
ax[1].plot(distance, analyticalCorrelation1D(distance, dimensions, 1), color="black", label="Analytical", linewidth=2)
ax[1].set_ylabel("C(r)", fontsize=14)

c, m, m2, m4 = runModel(dimensions, 0.5)

ax[2].plot(distance, c, color="black", linestyle="dotted", linewidth=2)
ax[2].plot(distance, analyticalCorrelation1D(distance, dimensions, 0.5), color="black", linewidth=2)
ax[2].set_ylabel("C(r)", fontsize=14)
ax[2].set_xlabel("Spin", fontsize=14)

fig.legend(ncol=2, frameon=false, loc="upper center", fontsize=18)
fig.savefig("2b_3.pdf")


L = 16
dimensions = L, L
T = LinRange(0.1, 5, 40)
mT = zeros(Float64, length(T))
mT2 = zeros(Float64, length(T))
for i in 1:length(T)
	c, m, m2, m4 = runModel(dimensions, T[i], nEpochs=1, nCycles_equil=2000, nCycles_meas=10000)
	mT[i] = m
	mT2[i] = m2
end

fig, ax = subplots(1, 1, figsize=(10, 8))

ax.plot(T, mT, color="black")
ax.set_xlabel("T/J", fontsize=14)
ax.set_ylabel("m", fontsize=14)
fig.savefig("2c.pdf")

fig, ax = subplots(1, 1, figsize=(10, 8))

ax.plot(T, mT2, color="black")
ax.set_xlabel("T/J", fontsize=14)
ax.set_ylabel("m", fontsize=14)
fig.savefig("2d.pdf")

=#
fig, ax = subplots(1, 1)

T = LinRange(2, 2.5, 30)
dims = [8, 16, 32]
for dim in dims
	mT = zeros(Float64, length(T))
	mT2 = zeros(Float64, length(T))
	mT4 = zeros(Float64, length(T))
	for i in 1:length(T)
		c, m, m2, m4 = runModel((dim, dim), T[i], nEpochs=1, nCycles_equil=2000, nCycles_meas=10000)
		mT[i] = m
		mT2[i] = m2
		mT4[i] = m4
	end
	ax.plot(T, mT4./mT2.^2, label="$dim", linewidth=2)
end
ax.set_xlabel("T/J", fontsize=14)
ax.set_ylabel("Gamma", fontsize=14)
ax.legend(frameon=false, loc="best", fontsize=18)
fig.savefig("2f.pdf")
