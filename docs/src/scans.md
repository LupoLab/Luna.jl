# Parameter scans
Luna comes with a flexible interface to run, save and process scans over any parameter or combination of parameters you can think of. Scans can be executed in several ways, which are defined via the various subtypes of `Scans.AbstractExec`:
- Local 

## Execution over SSH
Setup steps required:
- On the remote machine, add Julia to your path upon loading even over SSH: add `export PATH=/opt/julia-1.5.1/bin:$PATH` or similar to your `.bashrc` file **above** the usual check for interactive running.
- On Windows, install OpenSSH 8:
   - Follow [these instructions]( https://github.com/PowerShell/Win32-OpenSSH/wiki/Install-Win32-OpenSSH) to install the new version.
   - **Uninstall** OpenSSH via [Windows Features](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse) (this removes OpenSSH 7) so that Windows finds OpenSSH 8 instead.
- Set up 