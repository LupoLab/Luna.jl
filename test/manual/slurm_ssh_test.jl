# ---------------------------------------------------------------------------------------------
# Minimal SSHExec + SlurmExec live smoke test — MANUAL, submits real cluster jobs
# ---------------------------------------------------------------------------------------------
# This is NOT part of the automated test suite (it needs SSH access to the cluster and a
# Slurm installation, and it submits real jobs), which is why it lives in test/manual/.
# The queue, completion-marker, --maxpoints and jobscript-generation logic it exercises are
# all unit-tested in test/test_scans.jl; what only this script can verify is the genuine
# end-to-end path: scp to the cluster, sbatch submission, and the generated job script
# actually draining a queue on real compute nodes.
#
# Run it from your *local* machine (with the Luna project active):
#
#     julia --project=/Users/jt52/.julia/dev/Luna test/manual/slurm_ssh_test.jl
#
# What happens:
#   1. Locally, `runscan` sees the SSHExec, scp's THIS file to the cluster and runs it there.
#   2. On the cluster, the same file runs again; now gethostname() matches `remotehostname`,
#      so it falls through to the SlurmExec and submits a Slurm array job with `sbatch`.
#   3. With `instances=2` (the current production configuration), each array task runs two
#      respawning shell loops, each launching `julia <thisfile> --queue --maxpoints 1` — a
#      fresh single-point process per scan point. Because the points are tiny (<1 s each),
#      this stress-tests the respawn machinery hard: expect ~50 process launches whose cost
#      is dominated by Julia startup (~1 min each), i.e. roughly 10 minutes end to end.
#   4. The final respawn in each loop finds the queue drained, writes the completion marker
#      (`qfile_*.done`) and exits; trailing array tasks that start after completion must
#      exit immediately instead of re-running the scan.
#
# Success criteria: one numbered .h5 per scan point in scanoutput/ (50 files), a
# `qfile_*.done` marker next to the queue file in the Slurm workdir, all array tasks
# finished, and `sacct` showing no OOM or non-zero exits.

using Luna

# Two distinct addresses (they differ on dmog):
#   hostname       -> the address ssh/scp connect to (must be reachable from your laptop)
#   remotehostname -> what `Base.gethostname()` returns *on the login node* (how Luna knows it
#                     is already on the remote and should submit, rather than SSH onwards).
# If you ever change clusters, re-check the remote name with:
#     julia -e 'println(Base.gethostname())'
hostname       = "dmog.hw.ac.uk"
remotehostname = "login1.pri.dmog.alces.network"

# Subdirectory (always relative to $HOME on the cluster) where a timestamped scan folder is
# created. ~/sharedscratch lives under $HOME, so this lands the whole scan -- script, Slurm
# workdir, queue file, .done marker and all output -- under
#   ~/sharedscratch/lunascans/<timestamp>_slurm_instances_test/
subdir = "sharedscratch/lunascans"

# --- Cluster job layout ----------------------------------------------------------------------
# 3 array tasks x 2 one-shot instances = 6 concurrent single-point processes -- deliberately
# small (DMOG allows 10 running jobs / 110 cores / 1006G, and this is only a smoke test).
slurm = Scans.SlurmExec(@__FILE__, 3;    # 3 array tasks
                        instances=2,       # 2 respawning single-point processes per task
                        nthreads=1,        # each simulation single-threaded (keeps it light)
                        memory="16G",      # per task, shared by the 2 instances (8G hint each)
                        time="00:30:00")   # dmog requires --time; ample for ~10 min of respawns
# To smoke-test the persistent-worker mode instead, swap the layout for:
#   slurm = Scans.SlurmExec(@__FILE__, 3; procs=4, nthreads=1, memory="16G", time="00:30:00")
#
# Luna must be importable on the cluster via plain `ssh <host> julia` (SSHExec runs the script
# there with no --project), i.e. installed in the cluster's *default* Julia environment.
# `project=` only sets which env the array *tasks* use; it defaults to the env detected at
# submit time on the cluster.

exec = Scans.SSHExec(slurm, hostname, subdir; remotehostname=remotehostname)

# --- A deliberately light scan ---------------------------------------------------------------
# Short fibre + low energy + coarse grid => each point runs in well under a second, so the
# wall time is dominated by the per-point Julia startup -- the overhead the instances mode
# deliberately accepts in exchange for returning all memory to the OS after every point.
energies = collect(range(1e-6, 5e-6; length=50))

scan = Scan("slurm_instances_test", exec; energy=energies)

# Output is written next to this script on the cluster (one numbered .h5 per scan point).
outputdir = joinpath(@__DIR__, "scanoutput")

runscan(scan) do scanidx, energy
    prop_capillary(125e-6, 0.1, :He, 0.4;     # 10 cm capillary, 0.4 bar He -- very light
                   λ0=800e-9, τfwhm=10e-15, energy,
                   λlims=(300e-9, 2e-6), trange=256e-15,  # coarse grid
                   scan, scanidx, filepath=outputdir)
end
