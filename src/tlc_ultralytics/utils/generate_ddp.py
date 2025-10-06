from ultralytics.utils.dist import generate_ddp_command as ultralytics_generate_ddp_command


def generate_ddp_command(world_size, trainer) -> str:
    """Override the generate_ddp_command function to serialize the 3LC data required for DDP processes
    as part of the data argument. The override is temporary and therefore does not alter the original trainer state.
    """
    original_data = trainer.args.data
    trainer.args.data = trainer._serialize_state()
    command = ultralytics_generate_ddp_command(world_size, trainer)
    trainer.args.data = original_data
    return command
