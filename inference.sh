torchrun --nproc-per-node=1 \
         --master-port=12546  ./inference.py -p options/inference/DeMoE.yml -c ./models/DeMoE.pt -i ./images/inputs -t auto

# For manually selecting the deblurring task, add the -t argument before running, otherwise the model will select the best expert to be used.
# values for -t --> [auto, defocus, global_motion, synth_global_motion, local_motion, low_light]