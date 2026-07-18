# Env vars use ${X:-default} so a caller can export them (the values below are
# unchanged defaults). LAUNCHER lets a caller use torchrun instead of python3.
# Trailing "$@" lets a caller append extra config overrides (last one wins in the
# omegaconf dotlist merge). The recipe itself is untouched.
NAME=${NAME:-stage1_bolmo_1b}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-4096} \
DTYPE=${DTYPE:-float32} \
DATA_SOURCE=${DATA_SOURCE:-data_sources.txt} \
OLMO_ARCH=${OLMO_ARCH:-olmo2_1B_v2} \
OLMO_CKPT_PATH=${OLMO_CKPT_PATH:-/path/to/olmo2/ckpt} \
TRAIN_MODE=${TRAIN_MODE:-stage_1} \
LOCAL_MODEL_STYLE=${LOCAL_MODEL_STYLE:-hnet:xlstm} \
ADD_HASH_EMBEDDINGS=${ADD_HASH_EMBEDDINGS:-false} \
ADD_EXPANDED_EMBEDDINGS=${ADD_EXPANDED_EMBEDDINGS:-true} \
EMBEDDING_INIT_PATH=${EMBEDDING_INIT_PATH:-""} \
SAVE_FOLDER=${SAVE_FOLDER:-/path/to/save/folder/$NAME} \
${LAUNCHER:-python3} src/examples/bolmo/train_stage1.py $NAME \
    train_module.bolmo_config.losses=[local_encoder,ce,local_decoder,boundary] \
    train_module.bolmo_config.loss_weights=[1,1,1,4] \
    train_module.bolmo_config.div_fn=kl \
    train_module.bolmo_config.binarization_temp=5.0 \
    train_module.bolmo_config.use_oracle_patch_reps=true \
    train_module.bolmo_config.teacher_force_boundaries=true \
    train_module.bolmo_config.encoder_loss_lookahead=4 \
    train_module.bolmo_config.encoder_loss_no_lookahead_weight=0.0 \
    train_module.bolmo_config.encoder_loss_lookahead_weights=[0.0,0.0,0.0,4.0] \
    train_module.bolmo_config.do_alm_debiasing=true \
    train_module.bolmo_config.merge_boundary_loss=false \
    train_module.optim.weight_decay=0.1 \
    train_module.max_grad_norm=0.5 \
    train_module.optim.lr=7e-4 \
    model.block.attention.use_flash=true \
    model.local_encoder.n_layers=1 \
    model.local_decoder.n_layers=4 \
    model.local_decoder.hnet_smooth=false \
    model.local_decoder.hnet_modulate=false \
    model.local_encoder.boundary_predictor_lookahead=1 \
    model.local_decoder.add_in_projection=true \
    model.local_decoder.add_norm_onto_residual=false \
    model.local_decoder.add_projected_patch_residuals=false \
    model.local_encoder.block_config.feed_forward.hidden_size=2816 \
    model.local_decoder.block_config.feed_forward.hidden_size=2816 \
    model.local_encoder.d_model=2048 \
    model.local_decoder.d_model=2048 \
    data_loader.global_batch_size=786432 \
    train_module.rank_microbatch_size=49152 \
    trainer.callbacks.checkpointer.ephemeral_save_interval=1000 \
    trainer.callbacks.checkpointer.save_interval=75000 \
    trainer.callbacks.downstream_evaluator.eval_interval=75000 \
    trainer.max_duration.value=75000 \
    "$@"