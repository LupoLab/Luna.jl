# Running scans
There are several utilities available to easily run arbitrary parameter scans: in `Luna.Scans`: `@scaninit`, `@scanvar`, and `@scan`; and in `Luna.Output`: `@ScanHDF5Output`.

The first step is to write a standard Luna script to run a *single* simulation in the scan, e.g. for something involving a 30 fs, 1 μJ pulse:
```julia
# Lots of setup work...
gausspulse = let τ=30e-15, λ=λ0

end
in1 = (func=gausspulse, energy=1e-6)
inputs = (in1, )
# More setup work...
Luna.run(...) # run the simulation
```

Then, make the following changes:
+ Put `@scaninit [scanname]` at the very beginning of the script (after the `import` statements) e.g.
```julia
@scaninit "energy_duration_scan"
# Rest of the script
```

+ Create the arrays/ranges to be scanned over *in the script itself*, and annotate them with `@scanvar` e.g.
```julia
@scanvar energy = collect(range(1e-6, 10e-6, length=32))
@scanvar τ = (25:35)*1e-15
```

+ Wrap the main body of the script (**excluding the array definitions using `@scanvar`**) in a `@scan begin...end` block. In this block, use the interpolation syntax `$` to enter the scan variables wherever they need to be e.g.
```julia
# Copied from above
@scanvar energy = collect(range(1e-6, 10e-6, length=32))
@scanvar τ = (25:35)*1e-15

@scan begin
# Lots of setup work...
inputs = Fields.GaussField(λ0=λ0, τfwhm=$τ, energy=$energy)
# More setup work...
Luna.run(...) # run the simulation
end
```

   In the `@scan begin...end` block you also have access to `__SCANIDX__` which is a unique identifier for each point in the scan.
  
To run the scan, you run the script (say it's called `energy_duration_scan.jl`) from the command line. There are several options:

+ `$ julia energy_duration_scan.jl --local` simply runs the whole scan locally

+ `$ julia energy_duration_scan.jl --range [range]` runs the items given by `range`, which must evaluate to a valid `UnitRange`. The total number of items is `N1 * N2 * N3...` where each `Ni` is the length of one scan array annotated by `@scanvar`. The first `@scanvar` statement determines the innermost (fastest varying) loop, the second `@scanvar` statement the next loop and so forth. In our example above, this means `$ julia energy_duration_scan.jl --range 1:10` would run the first 10 (out of 32) energies for the first pulse duration (25 fs).

+ `$ julia energy_duration_scan.jl --batch idx,N` (note no spaces between comma and `N`) runs batch number `idx` out of a total of `N` batches. The batches are not sequential, e.g. for `N=4`, the indices run by the 4 batches are
```
idx=1 -> [1, 5, 9, ...]
idx=2 -> [2, 6, 10, ...]
idx=3 -> [3, 7, 11, ...]
idx=4 -> [4, 8, 12, ...]
```
This mode of running is mostly intended for parallel execution on clusters and workstations
   
+ `$ julia energy_duration_scan.jl --condor N` creates a `.sub` job script to submit via `condor_submit`, which will create `N` jobs which each run a batch using the `--batch` option. Note that for this to work, you need to run `julia energy_duration_scan.jl --condor N` **on the workstation** rather than your local machine.

+ `$ julia energy_duration_scan.jl --cirrus N` should only be run on cirrus or another cluster with PBS installed. It makes a jobscript (to be `qsub`ed) which submits `N` jobs, each of which executes one batch using the `--batch` option.


## Output for scans
For HDF5 output with automatic file naming for scans (e.g. `energy_duration_scan_00001.h5` etc), use `@ScanHDF5Output(...)` from `Luna.Output`. This only works within a `@scan begin...end` block and will automatically name your files according to the scan name given to `@scaninit` and the scan index. It will also save the name and current value of any variable scanned over in a group `"scanvars"` in the group `"meta"`. In our example above, `file["meta"]["scanvars"]` would contain `Dict{String,Any}("τ" => 2.5e-14,"energy" => 1.0e-7)`.
   
## Example script
For a fully worked-out example scan script, see `Luna/test/test_scan.jl`.

## Reference

```@docs
Scans.@scaninit name
```

```@docs
Scans.@scanvar
```

```@docs
Scans.@scan
```

```@docs
Scans.Scan
```

```@docs
Scans.makearray!(s::Scans.Scan)
```

```@docs
Scans.getval(s::Scans.Scan, var::Symbol, scanidx::Int)
```

```@docs
Scans.addvar!(s::Scans.Scan, var::Symbol, arr::AbstractArray)
```

```@docs
Scans.interpolate!(ex::Expr)
```

```@docs
Scans.chunks(a::AbstractArray, n::Int)
```

```@docs
length(s::Scans.Scan)
```