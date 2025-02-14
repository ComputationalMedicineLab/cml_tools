# Totally optional: add the following to .bashrc (or .zshrc, or whatever) to
# automatically turn on the Python environment and set variables. Or just
# source this file when needed.
micromamba activate cml_tools

export KMP_AFFINITY='granularity=fine,compact,1,0'
export KMP_BLOCKTIME=1
