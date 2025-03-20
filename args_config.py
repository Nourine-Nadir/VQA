PARSER_CONFIG = {
    'train_data_file':{
    'flags' : ('-tdfp', '--train_data_path'),
    'help': 'The train data file path (videos)',
    'type': str,
    'default':'/media/nadir/SSD/VQA Data/KVQ/Train/train_videos/'
    },

    'train_label_file':{
    'flags' : ('-trlfp', '--train_labels_path'),
    'help': 'The train label file path (MOS)',
    'type': str,
    'default':'/media/nadir/SSD/VQA Data/KVQ/Train/train_data.csv'
    },

    'train_data_dir': {
        'flags': ('-trdir', '--train_data_dir'),
        'help': 'The train data directory ',
        'type': str,
        'default': "/media/nadir/SSD/VQA Data/KVQ/extracted_features_rcnn/train_features"
    },

    'val_data_file':{
    'flags' : ('-vdfp', '--val_data_path'),
    'help': 'The validation data file path (videos)',
    'type': str,
    'default':'/media/nadir/SSD/VQA Data/KVQ/Validation/val_videos/'
    },

    'val_label_file':{
    'flags' : ('-vlfp', '--val_labels_path'),
    'help': 'The validation label file path (MOS)',
    'type': str,
    'default':'/media/nadir/SSD/VQA Data/KVQ/Validation/truth.csv'
    },

    'val_data_dir': {
            'flags': ('-vldir', '--val_data_dir'),
            'help': 'The val data directory ',
            'type': str,
            'default': "/media/nadir/SSD/VQA Data/KVQ/extracted_features_rcnn/val_features"
        },

    'test_data_file':{
        'flags' : ('-tsdir', '--test_data_path'),
        'help': 'The test data file path (videos)',
        'type': str,
        'default':'/media/nadir/SSD/VQA Data/KVQ/Test/test_videos/'
        },
    'test_example_csv':{
        'flags' : ('-tsexcsv', '--test_example_csv'),
        'help': 'The test data example submission csv',
        'type': str,
        'default': 'test_data.csv'
        },


    'test_data_dir': {
        'flags': ('-tsfd', '--test_data_dir'),
        'help': 'The test data directory ',
        'type': str,
        'default': "/media/nadir/SSD/VQA Data/KVQ/extracted_features_rcnn/test_features1"
    },

    "num_frames": {
        "flags": ("-nf", "--num_frames"),  # Can use proper tuple
        "help": "Number of frames for uniform video sizes",
        "type": int,  # Direct Python type
        "default": 20
    },

    "img_size": {
        "flags": ("-imgs", "--img_size"),  # Can use proper tuple
        "help": "Image size for uniform video shape",
        "type": int,  # Direct Python type
        "default": (540, 920)
    },

    "batch_size": {
        "flags": ("-bs", "--batch_size"),  # Can use proper tuple
        "help": "Batch size for training",
        "type": int,  # Direct Python type
        "default":int(8)
    },

    "epochs": {
        "flags": ("-ep", "--epochs"),
        "type": int,
        "help": "Number of epochs for training",
        "default": 100
    },

    "lr": {
            "flags": ("-lr", "--learning_rate"),
            "type": float,
            "help": "Learning Rate",
            "default": 1e-4
        },

    "verbose": {
        "flags": ("-v", "--verbose"),
        "help": "Enable verbose output",
        "action": "store_true"

    },


    "run_extraction": {
        "flags": ("-ruext", "--run_extraction"),
        "help": "Run Feature Extraction",
        "action": "store_true",
        "default": True

    },

    "run_training": {
        "flags": ("-rutr", "--run_training"),
        "help": "Run Training",
        "action": "store_true",
        "default": False
    },

    "run_testing": {
        "flags": ("-ruts", "--run_testing"),
        "help": "Run Testing",
        "action": "store_true",
        "default": False
    },

}