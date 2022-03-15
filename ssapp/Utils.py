from pathlib import Path


def get_model_save_path(type: str, name : str):
    main_dir = Path().cwd().parents[1]
    data_dir = main_dir / 'models'
    subdict_dir = data_dir / 'raw'
    dataset_dir = subdict_dir / 

    cut_dir = dataset_dir / 'cut_files'
    log_dir = dataset_dir / "log_files" 

    return cut_dir, log_dir