CFG = {
    "data": {
        "path": "Data_dir/fake_users.csv"
    },
    "test_data": {
        "path": "Data_dir/fake_users_test.csv"
    },
    "train": {
        "batch_size": 64,
        "epoches": 100,
        "metrics": ["accuracy"]
    },
    "model": {
        "layers": {
            "layer_1": 31,
            "layer_2": 64,
            "layer_3": 32,
            "layer_4": 25,
            "dropout": 0.4
        },
        "output": 1
    }
}