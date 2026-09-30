"""Pipeline step 06: fixed pre/post-learning decoder per mouse.

Trains, for each imaging mouse, a logistic-regression decoder separating
mapping responses of days -2/-1 from days +1/+2 (fast_learning.decoding),
oriented so that higher decision values mean post-learning.

Inputs:  mapping tensors (paths.tensor_dir), session metadata.
Outputs: <processed_dir>/decoding/decoder_weights.pkl    (scaler + classifier per mouse)
         <processed_dir>/decoding/classifier_weights.csv (per-cell weights)

Usage:
    python pipeline/06_decoder.py
"""

from fast_learning import decoding


if __name__ == '__main__':
    print("Loading mapping responses...")
    vectors_rew, vectors_nonrew, mice_rew, mice_nonrew = decoding.load_and_process_data()

    print("\nTraining and saving decoder weights...")
    decoding.train_and_save_decoder_weights(vectors_rew, vectors_nonrew, mice_rew, mice_nonrew)
