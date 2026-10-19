# LIBERO-Mem: use the libero-mem fork (read-only) instead of the shared LIBERO checkout. Source after env_g05.sh.
export LM=/export/home/kaziz/g05/libero_mem
export LIBERO_ROOT=$LM/libero-mem
export LIBERO_CONFIG_PATH=$LM/cfg
export PYTHONPATH=$LM/libero-mem:${PYTHONPATH:-}
