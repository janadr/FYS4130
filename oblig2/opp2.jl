using Random
using Plots


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

function constructCluster(randSpin, lattice, p)
	nElements = calculateLatticeElements(lattice)
	lattice[randSpin...] *= -1
	cluster = zeros(0)
	append!(cluster, randSpin)
	neighbours = findAllNeighbours(lattice, randSpin, dimensions)
	neighbours_copy = copy(neighbours)
	while length(neighbours) > 0
		for i in 1:length(neighbours)
			neighbours_copy = neighbours_copy[1:size(neighbours_copy, 1) .!= 1, :]
			if lattice[randSpin...] == lattice[neighbours[i]...]*(-1) && rand() <= p
				append!(cluster, neighbours[i])
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

function calculateCorrelation(lattice)
	dim = size(lattice, 1)
	r = zeros(dim)
	s = zeros(dim)
	for i in 1:dim
		r[i] += lattice[1, 1].*lattice[i, 1]
		s[i] += 1
	end
	CR = r./s
	return CR
end



Random.seed!(42)


nEpochs = 1
nCycles_equil = 10000
nCycles_meas = 1000

L = 16
dimensions = L
J = 1
T = 0.5
p = 1 - exp(-2*J*T)

CR = zeros(dimensions)
for epoch in 1:nEpochs
	lattice = rand(-1:2:1, dimensions)
	for cycle in 1:(nCycles_equil + nCycles_meas)
		randSpin = findRandomSpin(dimensions)
		lattice = constructCluster(randSpin, lattice, p)
		if cycle >= nCycles_equil
			CR .+= calculateCorrelation(lattice)
		end
	end
CR ./= nCycles_meas
end

plot(1:16, CR)
