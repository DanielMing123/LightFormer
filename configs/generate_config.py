import json
import os


def gen_config(out_path):
    config = dict(
        model_name = "LightFormerPredictor",
        image_num = 10,
        embed_dim = 256,
        num_heads = 8,
        num_sam_pts = 8,
        num_levels = 1,
        num_query = 1,
        log_every_n_steps = 10,
        out_class_num = 4,
        training = dict(
            sample_database_folder = [
                "/workspace/debug/prediction_ml_framework/data/Bosch_dataset/train"
            ],
            batch_size = 8,
            loader_worker_num = 20,
            epoch = 50
        ),
        validation = dict(
            sample_database_folder = "/workspace/debug/prediction_ml_framework/data/Bosch_dataset/test",
            batch_size = 8,
            loader_worker_num = 20,
            check_interval = 1.0,
            limit_batches = 1.0
        ),
        test = dict(
            sample_database_folder = "/workspace/debug/prediction_ml_framework/data/Bosch_dataset/test",
            batch_size = 1,
            loader_worker_num = 20,
            visualization = False,
            test_result_pkl_dir = "/workspace/debug/prediction_ml_framework/pred_res",
        ),
        optim = dict(
            init_lr = 0.0001,
            step_size = 3,
            step_factor = 0.5,
            gradient_clip_val = None,
            gradient_clip_algorithm = "norm"
        )
    )


    with open(os.path.join(out_path, "Light_Former_config.json"), "w") as f:
        json.dump(config, f)

if __name__ == "__main__":
    out_path = "/media/tao/Data_Use/Light_Former/configs"
    gen_config(out_path)