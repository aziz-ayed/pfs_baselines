for cfg in default default_uni default_ctranspath
do
  case "$cfg" in
    default)             prefix="optimus" ;;
    default_uni)         prefix="uni"     ;;
    default_ctranspath)  prefix="chief"   ;;
  esac

  for s in 0 1 2 3 4
  do
    torchrun --standalone --nproc_per_node=7 \
             train.py \
             --config "configs/${cfg}.yaml" \
             --seed "$s" \
             --run_name "${prefix}_attnmil_seed_${s}"
  done
done