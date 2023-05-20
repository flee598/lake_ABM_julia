## First go at an AGM of lake food web dynamics in Julia, using Agents.jl

### 3D model of a lake food web.

Overview:

1: generate a 3D lake environment from bathymetry data (currently csv)

2: define lake cell traits (e.g. resource type (pelagic vs littoral), resource growth rate)

3: populate lake with fish 3 species, trout (top predator), smelt and koaro (competitors that consume resources produced in cells)

4: define fish movement (random in 3D, but weighted to move towards specific resource (koaro == littoral, smelt == pelagic), trout wherever koaro and smelt are)

5: define fish eat (koaro = littoral resource preference, smelt pelagic resource preference)

6: define fish reproduce (all three species have a chance of reproducing during a 2 month window each year, can't reproduce outside of this)

7: define fish die (high chance of death after spawning for all 3 species, also if energy gets too low, also small random chance)

8: define cell behaviour - resource growth (follows discrete time logistic growth equation)

9: run model!


### To do
koaro/smelt predator avoidance - can do last
