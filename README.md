# MAGIA
This implementation is derived from the paper **MAGIA: SENSING PER-IMAGE SIGNALS FROM SINGLE-ROUND AVERAGED GRADIENTS FOR LABEL-INFERENCE-FREE GRADIENT INVERSION**.

'''

# Warm up to full batch over 150 iterations (recommended default)
python main_MAGIA.py --sel_mode linear_warmup --l_iter 150 --n_iter 300

'''
