## First go at an AGM of lake food web dynamics in Julia, using Agents.jl

### 3D model of a lake food web.

Overview:

1: generate a 3D lake environment from bathymetry data (currently csv)

2: define lake cell traits (e.g. resource type (pelagic vs littoral), resource growth rate)

3: populate lake with fish 3 species, trout (top predator), smelt and koaro (competitors that consume resources produced in cells)

4: define fish movement (random in 3D, but weighted to move towards specific resource)

5: define fish eat

6: define fish reproduce

7: define fish die

8: run model!



### To do
trout movement

koaro predation
trout predation

koaro reproduction
trout reproduction

koaro/smelt predator avoidance - can do last
