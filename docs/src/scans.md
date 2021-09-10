# Parameter scans
Luna comes with a flexible interface to run, save and process scans over any parameter or combination of parameters you can think of. A simple example can be found in `examples/simple_interface/scan.jl`, and we will go through it here. There are only a few necessary steps to run a parameter scan.

**First**, define the fixed parameters (those which are not being scanned over):
```julia
using Luna
import PyPlot: plt

a = 125e-6
flength = 3
gas = :HeJ
λ0 = 800e-9
τfwhm = 10e-15
λlims = (100e-9, 4e-6)
trange = 400e-15
```

**Second**, create the arrays which define your parameter scan. A simulation will be run at every possible combination (the Cartesian product) of these arrays. In this example we will do a pressure-energy scan:
```julia
energies = collect(range(50e-6, 200e-6; length=16))
pressures = collect(0.6:0.4:1.4)
```

**Third**, create a `Scan` which will store the scan arrays and define how the scan is executed. You must give this a name--in this case, `"pressure_energy_example"`. Scan variables like `energies` and `pressures` can be passed at construction time or added later:
```julia
scan = Scan("pressure_energy_example"; energy=energies)
addvariable!(scan, :pressure, pressures)
```

**Fourth**, run the scan! Here we want to store the output of the scan in a subdirectory of the directory containing the script we're running to avoid clutter. Passing `scan` and `scanidx` to `prop_capillary` will mean that our output files (one for each simulation) are automatically numbered and some information about the scan being run is also stored.
```julia
# @__DIR__ gives the directory of the current file
outputdir = joinpath(@__DIR__, "scanoutput")

runscan(scan) do scanidx, energy, pressure
    prop_capillary(a, flength, gas, pressure; λ0, τfwhm, energy,
                   trange=400e-15, scan, scanidx, filepath=outputdir)
end
```
Here, `runscan` uses the [do-block syntax](https://docs.julialang.org/en/v1/manual/functions/#Do-Block-Syntax-for-Function-Arguments), which wraps its body in a function. Here, this function takes three arguments: `scanidx, energy, pressure`. Generally, `scanidx` is always present and uniquely identifies a specific simulation in the scan (it simply runs from 1 to the number of simulations in the scan, `length(scan)`). The number of subsequent arguments is equal to the number of scan variables you have added to the `scan`. **Note** that the order in which you add the variables to the scan matters, and it must match the arguments in the `do` block. Alternatively, we could also have wrapped our scan in a function ourselves:
```julia
function runone(scanidx, energy, pressure)
   prop_capillary(a, flength, gas, pressure; λ0, τfwhm, energy,
                   trange=400e-15, scan, scanidx, filepath=outputdir)
end
runscan(runone, scan)
```
but the `do` block syntax is exactly equivalent to this and usually easier.

Running this script will simply run all the simulations, one after the other, in the current Julia session. To alter this behaviour, `Scan` also takes another argument, the execution mode `Scans.AbstractExec`. For example, to run only the first 10 items in the scan, we can use a `RangeExec`:
```julia
scan = Scan("pressure_energy_example", Scans.RangeExec(1:10); energy=energies) # note the second argument here
addvariable!(scan, :pressure, pressures)
outputdir = joinpath(@__DIR__, "scanoutput")

runscan(scan) do scanidx, energy, pressure
    prop_capillary(a, flength, gas, pressure; λ0, τfwhm, energy,
                   λlims, trange, scan, scanidx, filepath=outputdir)
end
```

Scans can be executed in several ways, which are defined via the various subtypes of `Scans.AbstractExec`:
- [`LocalExec`](@ref Scans.LocalExec): simply run the whole scan on the current machine in a `for` loop
- [`RangeExec`](@ref Scans.RangeExec): run a subsection of the scan as given by a `UnitRange` (e.g. `1:10` for the first 10 elements)
- [`BatchExec`](@ref Scans.BatchExec): divide the scan into batches and run a specific batch (can be used to balance load between processes)
- [`QueueExec`](@ref Scans.QueueExec): create a "queue file" which is used to balance load between several processes. This can be executed from multiple processes simultaneously. Alternatively, `QueueExec` can be made to spawn several subprocesses on the local machine which then use the queueing system to balance load between them.
- [`CondorExec`](@ref Scans.CondorExec): create a submission file (aka job file) for an HTCondor batch system running on the current machine and submit it, claiming a specified number of nodes, to execute the scan using a `QueueExec`.
- [`SSHExec`](@ref Scans.SSHExec): use one of the other `AbstractExec` types but first transfer the file to a remote host via SSH and then execute it. (**Note**: the remote machine must have Julia and Luna available with the same versions of both, and Julia must be available in a shell via the `julia` command.) For more details on how to set up execution over SSH, see [below](#execution-over-ssh).

## Saving scan output

## Processing scan output

## Execution over SSH
Setup steps required:
- On the remote machine, add Julia to your path upon loading even over SSH: add `export PATH=/opt/julia-1.5.1/bin:$PATH` or similar to your `.bashrc` file **above** the usual check for interactive running.
- On Windows, install OpenSSH 8:
   - Follow [these instructions]( https://github.com/PowerShell/Win32-OpenSSH/wiki/Install-Win32-OpenSSH) to install the new version.
   - **Uninstall** OpenSSH via [Windows Features](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse) (this removes OpenSSH 7) so that Windows finds OpenSSH 8 instead.
- Set up a public/private key pair to enable SSH login without entering a password.